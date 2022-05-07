from fun import *
from typing import TypedDict


class Imagem:
    nome         : str = None
    nome_path    : str = None
    w            : int = None # x
    h            : int = None # y
    ch           : int = None
    array        : np.ndarray = None
    barray       : np.ndarray = None
    r            : tuple = None
    g            : tuple = None
    b            : tuple = None
    
    def imag_up(self, nome : str) -> None:
        self.nome = nome
        self.nome_path = nome_path = os.getcwd()+"/imagem/"+self.nome
        self.array = np.array(cv2.imread(nome_path))
        self.h, self.w, self.ch = self.shape()
    
    def weight(self) -> int:
        return os.path.getsize(self.nome_path)
    
    def shape(self) -> tuple:
        return self.array.shape
    
    def set_normfor(self, nome : str, ext : str) -> None:
        self.nome = nome
        self.ext = ext
    
    

class File:
    nome         : str = None
    nome_path    : str = None
    array        : np.ndarray = None
    data         : str   = None
    r            : tuple = None
    g            : tuple = None
    b            : tuple = None
    
    def arq_up(self, nome : str) -> None:
        self.nome = nome
        self.nome_path = nome_path = os.getcwd()+"/" +self.nome
        with open(nome, 'r') as f:
            self.data = f.read()
            f.close()
            
    def weight(self):
        return os.path.getsize(self.nome_path)
    
    def set_normfor(self, nome : str, ext : str) -> None:
        self.nome = nome
        self.ext = ext