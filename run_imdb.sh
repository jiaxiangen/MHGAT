#for i in 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.20
#do
#for j in 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 
#do
 #python main.py  --gat-type max --gpu 1 --epochs 100 --num-classes 3 --num-rels 4  --num-layers 3 --hidden-size 64 --residual $i --in-drop 0.6 --attn-drop 0.6 --out-drop $j --lr 0.001  --negative-slope 0.2 --early-stop   --datasets IMDB
#done
#done

#for i in 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59
#do
#for j in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
#do
#python main.py  --gat-type sum --gpu 1 --epochs 100 --num-classes 2 --num-rels 4  --num-layers 3 --hidden-size 64 --residual $i --in-drop 0.6 --attn-drop 0.6 --out-drop  $j --lr 0.001  --negative-slope 0.2 --early-stop   --datasets   DOUBAN_929 
#done
#done
#for i in 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29
#do
#for j in   0.3 0.35 0.4 0.45  0.5 0.55 0.6 0.65 0.7 
#do
#for i in 0.5 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.60
#do
#python main.py  --gat-type sum --gpu 1 --epochs 100 --num-classes 3 --num-rels 4  --num-layers 3 --hidden-size 64 --residual $i  --in-drop 0.6 --attn-drop 0.6 --out-drop $j --lr 0.001 --negative-slope 0.2 --early-stop   --datasets AMAZON
#done
#done
python main.py  --gat-type sum --gpu 1 --epochs 100 --num-classes 3 --num-rels 4  --num-layers 3 --hidden-size 64 --residual 0.5  --in-drop 0.6 --attn-drop 0.6 --out-drop 0.6 --lr 0.001 --negative-slope 0.2 --early-stop   --datasets AMAZON
for i in   16 32 64 128  256  512
do
python main.py  --gat-type max --gpu 1 --epochs 100 --num-classes 3 --num-rels 4  --num-layers 3 --hidden-size 64   --out-size  $i --residual 0.1 --in-drop 0.6 --attn-drop  0.6  --out-drop 0.5 --lr 0.001 --negative-slope 0.2 --early-stop   --datasets IMDB
#python main.py  --gat-type max  --gpu 1 --epochs 100 --num-classes 3 --num-rels 4  --num-layers 3 --hidden-size 64   --out-size  $i  --residual 0.1 --in-drop 0.6 --attn-drop 0.6 --out-drop 0.3  --lr 0.001  --negative-slope 0.2 --early-stop   --datasets   DOUBAN_929 
done
for i in   16 32 64 128  256  512 
do
python main.py  --gat-type max --gpu 1 --epochs 100 --num-classes 3 --num-rels 4  --num-layers 3 --hidden-size 64   --out-size  $i --residual 0.1 --in-drop 0.6 --attn-drop  0.6  --out-drop 0.5 --lr 0.001 --negative-slope 0.2 --early-stop   --datasets IMDB
#python main.py  --gat-type max  --gpu 1 --epochs 100 --num-classes 3 --num-rels 4  --num-layers 3 --hidden-size 64   --out-size  $i  --residual 0.1 --in-drop 0.6 --attn-drop 0.6 --out-drop 0.3  --lr 0.001  --negative-slope 0.2 --early-stop   --datasets   DOUBAN_929 
done
for i in   16 32 64 128  256  512 
do
python main.py  --gat-type max --gpu 1 --epochs 100 --num-classes 3 --num-rels 4  --num-layers 3 --hidden-size 64   --out-size  $i --residual 0.1 --in-drop 0.6 --attn-drop  0.6  --out-drop 0.5 --lr 0.001 --negative-slope 0.2 --early-stop   --datasets IMDB
#python main.py  --gat-type max  --gpu 1 --epochs 100 --num-classes 3 --num-rels 4  --num-layers 3 --hidden-size 64   --out-size  $i  --residual 0.1 --in-drop 0.6 --attn-drop 0.6 --out-drop 0.3  --lr 0.001  --negative-slope 0.2 --early-stop   --datasets   DOUBAN_929 
done
for i in   16 32 64 128  256  512 
do
python main.py  --gat-type max --gpu 1 --epochs 100 --num-classes 3 --num-rels 4  --num-layers 3 --hidden-size 64   --out-size  $i --residual 0.1 --in-drop 0.6 --attn-drop  0.6  --out-drop 0.5 --lr 0.001 --negative-slope 0.2 --early-stop   --datasets IMDB
#python main.py  --gat-type max  --gpu 1 --epochs 100 --num-classes 3 --num-rels 4  --num-layers 3 --hidden-size 64   --out-size  $i  --residual 0.1 --in-drop 0.6 --attn-drop 0.6 --out-drop 0.3  --lr 0.001  --negative-slope 0.2 --early-stop   --datasets   DOUBAN_929 
done
for i in   16 32 64 128  256  512 
do
python main.py  --gat-type max --gpu 1 --epochs 100 --num-classes 3 --num-rels 4  --num-layers 3 --hidden-size 64   --out-size  $i --residual 0.1 --in-drop 0.6 --attn-drop  0.6  --out-drop 0.5 --lr 0.001 --negative-slope 0.2 --early-stop   --datasets IMDB
#python main.py  --gat-type max  --gpu 1 --epochs 100 --num-classes 3 --num-rels 4  --num-layers 3 --hidden-size 64   --out-size  $i  --residual 0.1 --in-drop 0.6 --attn-drop 0.6 --out-drop 0.3  --lr 0.001  --negative-slope 0.2 --early-stop   --datasets   DOUBAN_929 
done
for i in   16 32 64 128  256  512 
do
python main.py  --gat-type max --gpu 1 --epochs 100 --num-classes 3 --num-rels 4  --num-layers 3 --hidden-size 64   --out-size  $i --residual 0.1 --in-drop 0.6 --attn-drop  0.6  --out-drop 0.5 --lr 0.001 --negative-slope 0.2 --early-stop   --datasets IMDB
#python main.py  --gat-type max  --gpu 1 --epochs 100 --num-classes 3 --num-rels 4  --num-layers 3 --hidden-size 64   --out-size  $i  --residual 0.1 --in-drop 0.6 --attn-drop 0.6 --out-drop 0.3  --lr 0.001  --negative-slope 0.2 --early-stop   --datasets   DOUBAN_929 
done
for i in   16 32 64 128  256  512 
do
python main.py  --gat-type max --gpu 1 --epochs 100 --num-classes 3 --num-rels 4  --num-layers 3 --hidden-size 64   --out-size  $i --residual 0.1 --in-drop 0.6 --attn-drop  0.6  --out-drop 0.5 --lr 0.001 --negative-slope 0.2 --early-stop   --datasets IMDB
#python main.py  --gat-type max  --gpu 1 --epochs 100 --num-classes 3 --num-rels 4  --num-layers 3 --hidden-size 64   --out-size  $i  --residual 0.1 --in-drop 0.6 --attn-drop 0.6 --out-drop 0.3  --lr 0.001  --negative-slope 0.2 --early-stop   --datasets   DOUBAN_929 
done
for i in   16 32 64 128  256  512 
do
python main.py  --gat-type max --gpu 1 --epochs 100 --num-classes 3 --num-rels 4  --num-layers 3 --hidden-size 64   --out-size  $i --residual 0.1 --in-drop 0.6 --attn-drop  0.6  --out-drop 0.5 --lr 0.001 --negative-slope 0.2 --early-stop   --datasets IMDB
#python main.py  --gat-type max  --gpu 1 --epochs 100 --num-classes 3 --num-rels 4  --num-layers 3 --hidden-size 64   --out-size  $i  --residual 0.1 --in-drop 0.6 --attn-drop 0.6 --out-drop 0.3  --lr 0.001  --negative-slope 0.2 --early-stop   --datasets   DOUBAN_929 
done
for i in   16 32 64 128  256  512 
do
python main.py  --gat-type max --gpu 1 --epochs 100 --num-classes 3 --num-rels 4  --num-layers 3 --hidden-size 64   --out-size  $i --residual 0.1 --in-drop 0.6 --attn-drop  0.6  --out-drop 0.5 --lr 0.001 --negative-slope 0.2 --early-stop   --datasets IMDB
#python main.py  --gat-type max  --gpu 1 --epochs 100 --num-classes 3 --num-rels 4  --num-layers 3 --hidden-size 64   --out-size  $i  --residual 0.1 --in-drop 0.6 --attn-drop 0.6 --out-drop 0.3  --lr 0.001  --negative-slope 0.2 --early-stop   --datasets   DOUBAN_929 
done
for i in   16 32 64 128  256  512 
do
python main.py  --gat-type max --gpu 1 --epochs 100 --num-classes 3 --num-rels 4  --num-layers 3 --hidden-size 64   --out-size  $i --residual 0.1 --in-drop 0.6 --attn-drop  0.6  --out-drop 0.5 --lr 0.001 --negative-slope 0.2 --early-stop   --datasets IMDB
#python main.py  --gat-type max  --gpu 1 --epochs 100 --num-classes 3 --num-rels 4  --num-layers 3 --hidden-size 64   --out-size  $i  --residual 0.1 --in-drop 0.6 --attn-drop 0.6 --out-drop 0.3  --lr 0.001  --negative-slope 0.2 --early-stop   --datasets   DOUBAN_929 
done
