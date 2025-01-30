for n in {1..50}; do 
  python "main.py" -b 4 -bs 256 -s 1.16 -ds CIFAR100 -m mobilenetv2_x0_75 -n 10
done