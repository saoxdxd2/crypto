package main

import (
	"archive/zip"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"time"
)

const NUM_WORKERS = 8

func main() {
	if len(os.Args) < 4 {
		fmt.Println("Usage: go run fast_fetch.go <URL> <ZIP_PATH> <OUTPUT_DIR>")
		os.Exit(1)
	}

	url := os.Args[1]
	zipPath := os.Args[2]
	outDir := os.Args[3]

	fmt.Printf("[Go Fetcher] 🚀 Starting highly parallel download from %s\n", url)
	start := time.Now()

	err := downloadFileParallel(url, zipPath)
	if err != nil {
		fmt.Printf("[Go Fetcher] ❌ Download failed: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("[Go Fetcher] ✅ Download complete in %v\n", time.Since(start))
	
	fmt.Printf("[Go Fetcher] ⚡ Extracting %s to %s natively in Go...\n", zipPath, outDir)
	extractStart := time.Now()
	
	err = extractZip(zipPath, outDir)
	if err != nil {
		fmt.Printf("[Go Fetcher] ❌ Extraction failed: %v\n", err)
		os.Exit(1)
	}
	
	fmt.Printf("[Go Fetcher] ✅ Extraction complete in %v. Total time: %v\n", time.Since(extractStart), time.Since(start))
}

func downloadFileParallel(url string, dest string) error {
	// Get file size
	resp, err := http.Head(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	sizeStr := resp.Header.Get("Content-Length")
	if sizeStr == "" {
		// Fallback to sequential if server doesn't support Content-Length
		fmt.Println("[Go Fetcher] Server does not support Content-Length. Falling back to sequential download.")
		return downloadFileSequential(url, dest)
	}

	size, err := strconv.ParseInt(sizeStr, 10, 64)
	if err != nil {
		return err
	}

	// Create output file with exact size
	out, err := os.Create(dest)
	if err != nil {
		return err
	}
	out.Truncate(size)
	out.Close()

	chunkSize := size / NUM_WORKERS
	var wg sync.WaitGroup
	errCh := make(chan error, NUM_WORKERS)

	for i := 0; i < int(NUM_WORKERS); i++ {
		wg.Add(1)
		start := int64(i) * chunkSize
		end := start + chunkSize - 1
		if i == int(NUM_WORKERS)-1 {
			end = size - 1 // Last chunk takes the remainder
		}

		go func(workerID int, start, end int64) {
			defer wg.Done()
			err := downloadChunk(url, dest, start, end)
			if err != nil {
				errCh <- err
			}
		}(i, start, end)
	}

	wg.Wait()
	close(errCh)

	if len(errCh) > 0 {
		return <-errCh // return first error
	}
	return nil
}

func downloadChunk(url string, dest string, start, end int64) error {
	client := &http.Client{Timeout: 30 * time.Minute}
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return err
	}
	req.Header.Set("Range", fmt.Sprintf("bytes=%d-%d", start, end))

	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusPartialContent && resp.StatusCode != http.StatusOK {
		return fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}

	// Open file for writing at offset
	out, err := os.OpenFile(dest, os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	defer out.Close()

	_, err = out.Seek(start, 0)
	if err != nil {
		return err
	}

	_, err = io.Copy(out, resp.Body)
	return err
}

func downloadFileSequential(url string, dest string) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	out, err := os.Create(dest)
	if err != nil {
		return err
	}
	defer out.Close()

	_, err = io.Copy(out, resp.Body)
	return err
}

func extractZip(zipFile, destDir string) error {
	r, err := zip.OpenReader(zipFile)
	if err != nil {
		return err
	}
	defer r.Close()

	err = os.MkdirAll(destDir, 0755)
	if err != nil {
		return err
	}

	for _, f := range r.File {
		fpath := filepath.Join(destDir, f.Name)
		if f.FileInfo().IsDir() {
			os.MkdirAll(fpath, os.ModePerm)
			continue
		}

		if err = os.MkdirAll(filepath.Dir(fpath), os.ModePerm); err != nil {
			return err
		}

		outFile, err := os.OpenFile(fpath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, f.Mode())
		if err != nil {
			return err
		}

		rc, err := f.Open()
		if err != nil {
			outFile.Close()
			return err
		}

		_, err = io.Copy(outFile, rc)
		outFile.Close()
		rc.Close()

		if err != nil {
			return err
		}
	}
	return nil
}
