echo 'celebAHQ-16-32' &&
python demo_many_only_test_real.py --model $1 --method_backbone $2 --resolution_in 16,16 --resolution_out 32,32 --downsampling nn65 --gpu $3 &&
echo 'celebAHQ-32-64' &&
python demo_many_only_test_real.py --model $1 --method_backbone $2 --resolution_in 32,32 --resolution_out 64,64 --downsampling nn65 --gpu $3 &&
echo 'celebAHQ-64-128' &&
python demo_many_only_test_real.py --model $1 --method_backbone $2 --resolution_in 64,64 --resolution_out 128,128 --downsampling nn65 --gpu $3 &&

true
