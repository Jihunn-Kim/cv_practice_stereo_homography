# Figure 15
# python dp_stereo_matching.py -l ./source_images/Piano-l.png -r ./source_images/Piano-r.png -s ./dp_images/dp_piano.png -o 35 -p 5 -c 180 -m grad -rs 1
# python dp_stereo_matching.py -l ./source_images/Piano-l.png -r ./source_images/Piano-r.png -s ./dp_images/dp_piano.png -o 35 -p 5 -c 180 -m grad -rs 0.5
# python dp_stereo_matching.py -l ./source_images/Piano-l.png -r ./source_images/Piano-r.png -s ./dp_images/dp_piano.png -o 35 -p 5 -c 180 -m grad -rs 0.3
# python dp_stereo_matching.py -l ./source_images/Piano-l.png -r ./source_images/Piano-r.png -s ./dp_images/dp_piano.png -o 35 -p 5 -c 180 -m grad -rs 0.2

# Figure 16
python dp_stereo_matching.py -l ./source_images/Piano-l.png -r ./source_images/Piano-r.png -s ./dp_images/dp_piano.png -o 1000 -p 5 -c 64 -m sqr -rs 0.3 -fl -flt wmf
python dp_stereo_matching.py -l ./source_images/Piano-l.png -r ./source_images/Piano-r.png -s ./dp_images/dp_piano.png -o -2 -p 5 -c 64 -m ncc -rs 0.3 -fl -flt wmf
python dp_stereo_matching.py -l ./source_images/Piano-l.png -r ./source_images/Piano-r.png -s ./dp_images/dp_piano.png -o 35 -p 5 -c 64 -m grad -rs 0.3 -fl -flt wmf


