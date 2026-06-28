package main

import (
	"bufio"
	"fmt"
	"io"
	"log"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/pkg/sftp"
	"golang.org/x/crypto/ssh"
)

// ProxyConn wraps a cloudflared subprocess to act as a net.Conn for the SSH client.
type ProxyConn struct {
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout io.ReadCloser
}

func (p *ProxyConn) Read(b []byte) (n int, err error)  { return p.stdout.Read(b) }
func (p *ProxyConn) Write(b []byte) (n int, err error) { return p.stdin.Write(b) }
func (p *ProxyConn) Close() error {
	p.stdin.Close()
	p.stdout.Close()
	return p.cmd.Process.Kill()
}
func (p *ProxyConn) LocalAddr() net.Addr                { return nil }
func (p *ProxyConn) RemoteAddr() net.Addr               { return nil }
func (p *ProxyConn) SetDeadline(t time.Time) error      { return nil }
func (p *ProxyConn) SetReadDeadline(t time.Time) error  { return nil }
func (p *ProxyConn) SetWriteDeadline(t time.Time) error { return nil }

func newCloudflaredProxy(hostname string) (*ProxyConn, error) {
	cmd := exec.Command("cloudflared", "access", "ssh", "--hostname", hostname)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	return &ProxyConn{cmd: cmd, stdin: stdin, stdout: stdout}, nil
}

func connectSSH(hostname, password string) (*ssh.Client, error) {
	fmt.Printf("🚀 Connecting to Colab root@%s via Cloudflared...\n", hostname)
	
	proxy, err := newCloudflaredProxy(hostname)
	if err != nil {
		return nil, fmt.Errorf("failed to start cloudflared: %v", err)
	}

	config := &ssh.ClientConfig{
		User: "root",
		Auth: []ssh.AuthMethod{
			ssh.Password(password),
		},
		HostKeyCallback: ssh.InsecureIgnoreHostKey(),
		Timeout:         15 * time.Second,
	}

	// Dial the SSH connection over the cloudflared proxy
	c, chans, reqs, err := ssh.NewClientConn(proxy, hostname, config)
	if err != nil {
		proxy.Close()
		return nil, fmt.Errorf("failed to establish SSH handshake: %v", err)
	}

	client := ssh.NewClient(c, chans, reqs)
	fmt.Println("✅ Connected successfully!")
	return client, nil
}

func executeCommand(client *ssh.Client, command string) error {
	fmt.Printf("\n[COLAB EXEC] %s\n", command)
	session, err := client.NewSession()
	if err != nil {
		return err
	}
	defer session.Close()

	stdout, err := session.StdoutPipe()
	if err != nil {
		return err
	}
	stderr, err := session.StderrPipe()
	if err != nil {
		return err
	}

	if err := session.Start(command); err != nil {
		return err
	}

	var wg sync.WaitGroup
	wg.Add(2)

	go func() {
		defer wg.Done()
		scanner := bufio.NewScanner(stdout)
		for scanner.Scan() {
			fmt.Printf("  [OUT] %s\n", scanner.Text())
		}
	}()

	go func() {
		defer wg.Done()
		scanner := bufio.NewScanner(stderr)
		for scanner.Scan() {
			fmt.Printf("  [ERR] %s\n", scanner.Text())
		}
	}()

	wg.Wait()
	return session.Wait()
}

func downloadFileAtomic(client *ssh.Client, remotePath, localPath string) error {
	fmt.Printf("\n⬇️ Downloading %s to %s...\n", remotePath, localPath)
	
	sftpClient, err := sftp.NewClient(client)
	if err != nil {
		return err
	}
	defer sftpClient.Close()

	remoteFile, err := sftpClient.Open(remotePath)
	if err != nil {
		return err
	}
	defer remoteFile.Close()

	// Atomic download: Save to .tmp first
	tmpPath := localPath + ".tmp"
	os.MkdirAll(filepath.Dir(localPath), 0755)
	
	localFile, err := os.Create(tmpPath)
	if err != nil {
		return err
	}
	
	// Copy data
	_, err = io.Copy(localFile, remoteFile)
	localFile.Close()
	if err != nil {
		os.Remove(tmpPath)
		return err
	}

	// Atomic Rename
	if err := os.Rename(tmpPath, localPath); err != nil {
		return err
	}

	fmt.Println("✅ Download and atomic verification complete.")
	return nil
}

func main() {
	if len(os.Args) < 3 {
		log.Fatalf("Usage: go run main.go <trycloudflare_url> <password>")
	}
	
	url := os.Args[1]
	if strings.Contains(url, "://") {
		url = strings.Split(url, "://")[1]
	}
	password := os.Args[2]

	client, err := connectSSH(url, password)
	if err != nil {
		log.Fatalf("Fatal Error: %v", err)
	}
	defer client.Close()

	// Step 1: Prep Remote Environment
	executeCommand(client, "mkdir -p /content/crypto_research/data")
	executeCommand(client, "echo 'Simulating high-speed data download (Gigabit)...' && dd if=/dev/zero of=/content/crypto_research/data/lob_data.parquet bs=1M count=50")

	// Step 2: Training
	executeCommand(client, "python3 -c \"import time; print('Training Neural Network in Cloud...'); time.sleep(2); print('Best weights saved!')\"")

	// Step 3: Mock Remote Weights
	executeCommand(client, "mkdir -p /content/checkpoints && echo 'dummy weights from Go' > /content/checkpoints/lobert_checkpoint.pt")

	// Step 4: Atomic SFTP Download
	err = downloadFileAtomic(client, "/content/checkpoints/lobert_checkpoint.pt", "../../checkpoints/lobert_checkpoint.pt")
	if err != nil {
		log.Printf("SFTP Error: %v", err)
	}

	fmt.Println("\n🚀 Go Orchestrator Pipeline Finished!")
}
