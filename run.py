from fun import *

## ['run.py', <mode> <plano rgb> <imagem> <file>

if __name__ == "__main__":
    #print(sys.argv[3][-4:])
    if sys.argv[3][-4:]==".png":
        func[sys.argv[1]](sys.argv[3], sys.argv[4], func[sys.argv[2]])
        