# Figure 16
python dp_stereo_matching.py -l ./source_images/Pipes-l.png -r ./source_images/Pipes-r.png -s ./dp_images/dp_Pipes.png -o 1000 -p 5 -c 64 -m sqr -rs 0.3 -fl -flt wmf
python dp_stereo_matching.py -l ./source_images/Pipes-l.png -r ./source_images/Pipes-r.png -s ./dp_images/dp_Pipes.png -o -2 -p 5 -c 64 -m ncc -rs 0.3 -fl -flt wmf
python dp_stereo_matching.py -l ./source_images/Pipes-l.png -r ./source_images/Pipes-r.png -s ./dp_images/dp_Pipes.png -o 35 -p 5 -c 64 -m grad -rs 0.3 -fl -flt wmf


