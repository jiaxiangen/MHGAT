for i in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
for j in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
python main.py  --gat-type max  --gpu 1 --epochs 100 --num-classes 2 --num-rels 4  --num-layers 3 --hidden-size 64 --residual $i --in-drop 0.6 --attn-drop 0.6 --out-drop  $j --lr 0.001  --negative-slope 0.2 --early-stop   --datasets  DOUBAN_929 
done
done