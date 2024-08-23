import os
import cv2
import matplotlib.pyplot as plt
from modules.parallax import ParallaxGenerator
import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', help='ステレオグラムにしたい画像のパス(path to the image file you want to stereogram)')
    parser.add_argument('-o','--outpath', help='ステレオグラムを保存するディレクトリ名またはファイル名(directory-name or file-name where the stereograms will be stored)')
    parser.add_argument('-mpx','--max_parallax', default=15, help='視差を作り出すズレの最大量px(maximum amount px of displacement that creates parallax)')
    parser.add_argument('--downscale', type=float, help='画像を縮小して実行時間を短縮')
    parser.add_argument('--cross', action='store_true', help='交差法による裸眼立体視')
    parser.add_argument('--pyplot', action='store_true', help='pyplotによる画像表示')
    parser.add_argument('--split', action='store_true', help='ステレオグラムの左右を別々に保存する')
    args = parser.parse_args()

    img = cv2.imread(args.filepath)
    assert img is not None, f"can not open the file({args.filepath})"
    if args.downscale is not None:
        img = cv2.resize(img, None, fx=args.downscale, fy=args.downscale)

    pg = ParallaxGenerator(img, max_parallax=args.max_parallax) # ステレオグラム生成器を作成
    pg.generate() # ステレオグラム化の実行

    if args.pyplot:
        dst = pg.getStereogram(swapLR=args.cross) # 見やすいように整形したステレオグラムを作成
        plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
        plt.title("Stereogram")
        plt.axis('off')
        plt.show()

    if args.outpath is not None:
        if os.path.isdir(args.outpath):
            outpath = os.path.join(args.outpath, os.path.splitext(os.path.basename(args.filepath))[0])
        else:
            outpath = os.path.splitext(args.outpath)[0]
        if args.split:
            L,R = pg.getStereogram(swapLR=False, split=True)
            # L_path = outpath + '_L.png'
            # R_path = outpath + '_R.png'
            outpath = [outpath + '_L.png', outpath + '_R.png']
            cv2.imwrite(outpath[0], L)
            cv2.imwrite(outpath[1], R)
        else:
            dst = pg.getStereogram(swapLR=False, split=False)
            outpath += '.png'
            cv2.imwrite(outpath, dst)
        print(f"saved at {outpath}")
