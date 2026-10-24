"""Security and conformance regression for CRF ZIP assets."""
import hashlib
import io
import json
import struct
import unittest
import warnings
import zipfile
import zlib
from pathlib import Path

from coherent_diffusion.crf.assets import load_crf_zip
from coherent_diffusion.crf.validation import CRFError

BASE = json.loads(Path("examples/one_sun.crf.json").read_text())
def png(width=4,height=3):
    def chunk(k,v):
        return struct.pack(">I",len(v))+k+v+struct.pack(">I",zlib.crc32(k+v))
    ihdr = struct.pack(">IIBBBBB",width,height,8,0,0,0,0)
    pixels = (b"\x00"+b"\xff"*width)*height
    return b"\x89PNG\r\n\x1a\n"+chunk(b"IHDR",ihdr)+chunk(b"IDAT",zlib.compress(pixels))+chunk(b"IEND",b"")

def scene(data, **attrs):
    d=json.loads(json.dumps(BASE))
    d["regions"][-1]["geometry"]={"type":"mask","asset_id":"paint"}
    d["assets"]=[{"id":"paint","path":"masks/paint.png","sha256":hashlib.sha256(data).hexdigest(),
                  "media_type":"image/png","width":4,"height":3}]
    d["assets"][0].update(attrs)
    return d

def make_zip(doc, files, *, extra=None):
    f=io.BytesIO()
    with zipfile.ZipFile(f,"w",compression=zipfile.ZIP_STORED) as z:
        z.writestr("composition.json",json.dumps(doc))
        z.writestr("manifest.json",json.dumps({"format":"coherent-regions-archive","version":1}))
        for name,bytes_ in files.items():
            z.writestr(name,bytes_)
        for name,bytes_ in extra or []:
            z.writestr(name,bytes_)
    return f.getvalue()

class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.image=png()
        self.document=scene(self.image)
        self.files={"masks/paint.png":self.image}
    def load(self,**kwargs):
        return load_crf_zip(make_zip(kwargs.get("doc",self.document),kwargs.get("files",self.files),extra=kwargs.get("extra")))
    def test_valid_verified_mask(self):
        r=self.load()
        self.assertEqual(r.assets["paint"],self.image)
        self.assertEqual(r.report.document["regions"][-1]["geometry"]["type"],"mask")
    def test_wrong_hash(self):
        with self.assertRaisesRegex(CRFError,"checksum"):self.load(files={"masks/paint.png":self.image+b"x"})
    def test_missing_member(self):
        with self.assertRaisesRegex(CRFError,"declared assets"):self.load(files={})
    def test_extra_member(self):
        with self.assertRaisesRegex(CRFError,"declared assets"):self.load(extra=[("masks/other.png",self.image)])
    def test_duplicate(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",UserWarning)
            with self.assertRaisesRegex(CRFError,"Duplicate"):self.load(extra=[("masks/paint.png",self.image)])
    def test_path_traversal(self):
        with self.assertRaisesRegex(CRFError,"Unsafe"):self.load(extra=[("../escape",b"bad")])
    def test_absolute_path(self):
        with self.assertRaisesRegex(CRFError,"Unsafe"):self.load(extra=[("/tmp/payload",b"bad")])
    def test_size_mismatch(self):
        with self.assertRaisesRegex(CRFError,"dimensions"):self.load(doc=scene(self.image,width=99))
    def test_corrupt_header(self):
        d=b"BAD"*15
        with self.assertRaisesRegex(CRFError,"PNG"):self.load(doc=scene(d),files={"masks/paint.png":d})
    def test_archive_size_limit(self):
        with self.assertRaisesRegex(CRFError,"24 MiB"):load_crf_zip(b"x"*(24*1024*1024+1))
    def test_unexpected_file(self):
        with self.assertRaisesRegex(CRFError,"Unsafe"):self.load(extra=[("evil.txt",b"txt")])

if __name__=="__main__":
    unittest.main()
