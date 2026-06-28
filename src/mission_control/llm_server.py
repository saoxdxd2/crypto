import json
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# Setup simple logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LLM_Server")

class LLMServer:
    def __init__(self):
        self.llm = None
        self.model_path = Path("data/models/Qwen3-0.6B-Q4_K_M.gguf")

    def load_model(self):
        if not self.model_path.exists():
            logger.error(f"Model not found at {self.model_path}")
            return False
            
        try:
            from llama_cpp import Llama
            logger.info(f"Loading {self.model_path.name} into RAM (Shared Instance)...")
            self.llm = Llama(
                model_path=str(self.model_path),
                n_ctx=2048,
                n_threads=4,
                verbose=False
            )
            logger.info("Shared LLM successfully loaded into memory!")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def generate(self, prompt: str, max_tokens: int = 150, temperature: float = 0.3) -> str:
        if not self.llm:
            return "Error: LLM not loaded."
        
        try:
            res = self.llm(prompt, max_tokens=max_tokens, temperature=temperature, stop=["<|im_end|>"])
            text = res["choices"][0]["text"].strip()
            
            # Clean up markdown JSON blocks if present
            if text.startswith("```json"):
                text = text.replace("```json", "").replace("```", "").strip()
            return text
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"Error during generation: {e}"

# Global instance
server_instance = LLMServer()

class RequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/generate':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                prompt = data.get('prompt', '')
                max_tokens = data.get('max_tokens', 150)
                temperature = data.get('temperature', 0.3)
                
                response_text = server_instance.generate(prompt, max_tokens, temperature)
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"response": response_text}).encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

def run_server(port=5001):
    # Try loading the model first
    if not server_instance.load_model():
        logger.warning("Starting server WITHOUT model. Endpoints will return errors until model is downloaded.")
        
    server_address = ('', port)
    httpd = HTTPServer(server_address, RequestHandler)
    logger.info(f"Shared LLM Server running on port {port}...")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    logger.info("LLM Server stopped.")

if __name__ == '__main__':
    run_server()
