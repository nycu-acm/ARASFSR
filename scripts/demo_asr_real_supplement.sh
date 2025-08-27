echo 'celebAHQ-16-64' &&
python demo_many_only_test_real.py --model $1 --method_backbone $2 --resolution_in 16,16 --resolution_out 64,64 --downsampling nn65 --gpu $3 &&
echo 'celebAHQ-16-128' &&
python demo_many_only_test_real.py --model $1 --method_backbone $2 --resolution_in 16,16 --resolution_out 128,128 --downsampling nn65 --gpu $3 &&
echo 'celebAHQ-16-256' &&
python demo_many_only_test_real.py --model $1 --method_backbone $2 --resolution_in 16,16 --resolution_out 256,256 --downsampling nn65 --gpu $3 &&
echo 'celebAHQ-48-128' &&
python demo_many_only_test_real.py --model $1 --method_backbone $2 --resolution_in 48,48 --resolution_out 128,128 --downsampling nn65 --gpu $3 &&
echo 'celebAHQ-48-256' &&
python demo_many_only_test_real.py --model $1 --method_backbone $2 --resolution_in 48,48 --resolution_out 256,256 --downsampling nn65 --gpu $3 &&
echo 'celebAHQ-48-512' &&
python demo_many_only_test_real.py --model $1 --method_backbone $2 --resolution_in 48,48 --resolution_out 512,512 --downsampling nn65 --gpu $3 &&

true
