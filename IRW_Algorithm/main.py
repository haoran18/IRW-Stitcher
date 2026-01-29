from option import Options
from IRW_Stitcher import IRW_Stitching

if __name__ == '__main__':
    opt = Options().parse()
    result = IRW_Stitching(opt)
