import tempfile,threading,unittest,urllib.request,urllib.error
from pathlib import Path
from http.server import ThreadingHTTPServer
from server import Handler

class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();root=Path(cls.temp.name);(root/'test.mp4').write_bytes(b'0123456789')
        cls.server=ThreadingHTTPServer(('127.0.0.1',0),Handler);cls.server.data=root
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start()
        cls.base=f'http://127.0.0.1:{cls.server.server_port}'
    @classmethod
    def tearDownClass(cls):cls.server.shutdown();cls.server.server_close();cls.temp.cleanup()
    def test_video_byte_ranges(self):
        for header,body,content_range in [('bytes=2-5',b'2345','bytes 2-5/10'),('bytes=-3',b'789','bytes 7-9/10')]:
            with urllib.request.urlopen(urllib.request.Request(self.base+'/media/test.mp4',headers={'Range':header})) as r:
                self.assertEqual(r.status,206);self.assertEqual(r.read(),body);self.assertEqual(r.headers['Content-Range'],content_range)
    def test_invalid_range(self):
        with self.assertRaises(urllib.error.HTTPError) as e:urllib.request.urlopen(urllib.request.Request(self.base+'/media/test.mp4',headers={'Range':'bytes=20-'}))
        self.assertEqual(e.exception.code,416)
    def test_path_stays_in_data_directory(self):
        with self.assertRaises(urllib.error.HTTPError) as e:urllib.request.urlopen(self.base+'/media/%2e%2e/server.py')
        self.assertEqual(e.exception.code,404)

if __name__=='__main__':unittest.main()
