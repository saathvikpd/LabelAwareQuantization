for n in {2..20}; do 
  for sim in {0..1}; do 
    python "main.py" -b 4 -bs 256 -s 1.16 -ds CIFAR100 -m resnet_20 -sim $sim -n $n
  done
done