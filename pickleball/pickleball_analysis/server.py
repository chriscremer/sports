"""Local viewer with byte-range video serving. All generated data stays outside code."""
import argparse,json,mimetypes,re
from pathlib import Path
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from urllib.parse import unquote,urlsplit

WEB=Path(__file__).parent/'web'
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url=unquote(urlsplit(self.path).path)
        if url=='/api/clips':
            clips=[]
            for p in sorted(self.server.data.joinpath('analysis').glob('rally-*.json')):
                d=json.loads(p.read_text()); clips.append({'file':p.name,**d['clip']})
            body=json.dumps(clips).encode();self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body);return
        root=self.server.data if url.startswith('/media/') else WEB
        path=(root/(url[7:] if url.startswith('/media/') else url.lstrip('/') or 'index.html')).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file(): self.send_error(404);return
        size=path.stat().st_size;start,end=0,size-1; partial=False
        request_range=self.headers.get('Range')
        if request_range:
            match=re.fullmatch(r'bytes=(\d*)-(\d*)',request_range)
            if not match or not any(match.groups()): self.send_error(416);return
            a,b=match.groups()
            if a: start=int(a);end=min(int(b),end) if b else end
            else: start=max(0,size-int(b))
            if start>end or start>=size:
                self.send_response(416);self.send_header('Content-Range',f'bytes */{size}');self.end_headers();return
            partial=True
        self.send_response(206 if partial else 200)
        mime=mimetypes.guess_type(path)[0] or 'application/octet-stream'
        self.send_header('Content-Type',mime);self.send_header('Accept-Ranges','bytes');self.send_header('Cache-Control','no-cache');self.send_header('Content-Length',str(end-start+1))
        if partial:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
        self.end_headers()
        try:
            with path.open('rb') as f:
                f.seek(start);remaining=end-start+1
                while remaining:
                    chunk=f.read(min(1024*1024,remaining))
                    if not chunk:break
                    self.wfile.write(chunk);remaining-=len(chunk)
        except (BrokenPipeError,ConnectionResetError):pass

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8765);p.add_argument('--data',type=Path,default=Path('/Users/chriscremer/Downloads/pickleball_video_analysis'));args=p.parse_args()
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler);server.data=args.data
    print(f'Pickleball viewer: http://127.0.0.1:{args.port}',flush=True);server.serve_forever()
