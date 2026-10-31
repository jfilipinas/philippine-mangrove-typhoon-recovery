"""Seekable HTTP range reader using the environment's trusted TLS configuration.
Avoids the separate CA store in GDAL's bundled libcurl; TLS stays verified.
"""
import io, requests, math, os, threading, time
os.environ.setdefault('GDAL_CURL_CA_BUNDLE',os.environ.get('SSL_CERT_FILE','/etc/ssl/certs/ca-certificates.crt'))
os.environ.setdefault('GDAL_DISABLE_READDIR_ON_OPEN','EMPTY_DIR')
os.environ.setdefault('GDAL_HTTP_TIMEOUT','20')
os.environ.setdefault('CPL_VSIL_CURL_ALLOWED_EXTENSIONS','.tif,.tiff')
import rasterio
from contextlib import contextmanager

class HTTPRangeFile(io.RawIOBase):
    def __init__(self,url,block=262144):
        super().__init__();self.url=url;self.block=block;self.pos=0;self.cache={};self.session=requests.Session();self.size=None
        self._block(0)
    def _block(self,i):
        if i in self.cache:return self.cache[i]
        start=i*self.block;end=start+self.block-1
        for attempt in range(3):
            r=self.session.get(self.url,headers={'Range':f'bytes={start}-{end}'},timeout=25)
            if r.status_code not in (429,500,502,503,504):break
        r.raise_for_status()
        if r.status_code!=206:raise IOError(f'Range request not honoured: {r.status_code}')
        self.size=int(r.headers['Content-Range'].split('/')[-1]);self.cache[i]=r.content
        return r.content
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
        return self.pos
    def read(self,n=-1):
        if n<0:n=self.size-self.pos
        end=min(self.pos+n,self.size);out=[]
        while self.pos<end:
            i=self.pos//self.block;b=self._block(i);off=self.pos%self.block;take=min(end-self.pos,len(b)-off)
            if take<=0:break
            out.append(b[off:off+take]);self.pos+=take
        return b''.join(out)
    def readinto(self,b):
        d=self.read(len(b));b[:len(d)]=d;return len(d)
    def close(self):
        if hasattr(self,'session'):self.session.close()
        super().close()

_tokens={};_lock=threading.Lock()
def sign(href):
    from urllib.parse import urlsplit
    key='/'.join(href.split('/')[:4])
    with _lock:
        if key in _tokens and time.time()-_tokens[key][0]<1800:return href+'?'+_tokens[key][1]
        r=requests.get('https://planetarycomputer.microsoft.com/api/sas/v1/sign',params={'href':href},timeout=30);r.raise_for_status();signed=r.json()['href']
        _tokens[key]=(time.time(),urlsplit(signed).query)
    return signed

@contextmanager
def open_remote(href):
    url=sign(href)
    with rasterio.Env(GDAL_CURL_CA_BUNDLE=os.environ['GDAL_CURL_CA_BUNDLE'],GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',GDAL_HTTP_TIMEOUT=20):
        with rasterio.open(url) as ds:yield ds