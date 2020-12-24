# Figure 3
# python dp_stereo_matching.py -l ./source_images/tsukuba-l.png -r ./source_images/tsukuba-r.png -s ./dp_images/dp_tsukuba.png -o 1000 -p 5 -c 16 -m sqr -rs 1
# python dp_stereo_matching.py -l ./source_images/tsukuba-l.png -r ./source_images/tsukuba-r.png -s ./dp_images/dp_tsukuba.png -o 1000 -p 5 -c 16 -m abs -rs 1

# Figure 4
# python dp_stereo_matching.py -l ./source_images/tsukuba-l.png -r ./source_images/tsukuba-r.png -s ./dp_images/dp_tsukuba.png -o 2200 -p 7 -c 16 -m sqr -rs 1
# python dp_stereo_matching.py -l ./source_images/tsukuba-l.png -r ./source_images/tsukuba-r.png -s ./dp_images/dp_tsukuba.png -o 4000 -p 9 -c 16 -m sqr -rs 1

# Figure 5
# python dp_stereo_matching.py -l ./source_images/tsukuba-l.png -r ./source_images/tsukuba-r.png -s ./dp_images/dp_tsukuba.png -o 500 -p 5 -c 16 -m sqr -rs 1
# python dp_stereo_matching.py -l ./source_images/tsukuba-l.png -r ./source_images/tsukuba-r.png -s ./dp_images/dp_tsukuba.png -o 1500 -p 5 -c 16 -m sqr -rs 1

# Figure 6
# python dp_stereo_matching.py -l ./source_images/tsukuba-l.png -r ./source_images/tsukuba-r.png -s ./dp_images/dp_tsukuba.png -o 1000 -p 5 -c 16 -m sqr -rs 1 -f -ft gaussian -cs 3
# python dp_stereo_matching.py -l ./source_images/tsukuba-l.png -r ./source_images/tsukuba-r.png -s ./dp_images/dp_tsukuba.png -o 1000 -p 5 -c 16 -m sqr -rs 1 -f -ft gaussian -cs 5
# python dp_stereo_matching.py -l ./source_images/tsukuba-l.png -r ./source_images/tsukuba-r.png -s ./dp_images/dp_tsukuba.png -o 1000 -p 5 -c 16 -m sqr -rs 1 -f -ft gaussian -cs 7

# Figure 7
# python dp_stereo_matching.py -l ./source_images/tsukuba-l.png -r ./source_images/tsukuba-r.png -s ./dp_images/dp_tsukuba.png -o 1000 -p 5 -c 16 -m sqr -rs 1 -fl -flt simple

# Figure 8
# python dp_stereo_matching.py -l ./source_images/tsukuba-l.png -r ./source_images/tsukuba-r.png -s ./dp_images/dp_tsukuba.png -o 1000 -p 5 -c 16 -m sqr -rs 1 -f -ft gaussian -cs 3 -fl -flt simple

# Figure 12
# python dp_stereo_matching.py -l ./source_images/tsukuba-l.png -r ./source_images/tsukuba-r.png -s ./dp_images/dp_tsukuba.png -o 1000 -p 5 -c 16 -m sqr -rs 1 -f -ft bilateral
# python dp_stereo_matching.py -l ./source_images/tsukuba-l.png -r ./source_images/tsukuba-r.png -s ./dp_images/dp_tsukuba.png -o 1000 -p 5 -c 16 -m sqr -rs 1 -f -ft guided

# Figure 13
# python dp_stereo_matching.py -l ./source_images/tsukuba-l.png -r ./source_images/tsukuba-r.png -s ./dp_images/dp_tsukuba.png -o -2 -p 5 -c 16 -m ncc -rs 1
# python dp_stereo_matching.py -l ./source_images/tsukuba-l.png -r ./source_images/tsukuba-r.png -s ./dp_images/dp_tsukuba.png -o 35 -p 5 -c 16 -m grad -rs 1
# python dp_stereo_matching.py -l ./source_images/tsukuba-l.png -r ./source_images/tsukuba-r.png -s ./dp_images/dp_tsukuba.png -o 1 -p 5 -c 16 -m census -rs 1

# Figure 14
python dp_stereo_matching.py -l ./source_images/tsukuba-l.png -r ./source_images/tsukuba-r.png -s ./dp_images/dp_tsukuba.png -o 1000 -p 5 -c 16 -m sqr -rs 1 -f -ft gaussian -cs 3 -fl -flt wmf
python dp_stereo_matching.py -l ./source_images/tsukuba-l.png -r ./source_images/tsukuba-r.png -s ./dp_images/dp_tsukuba.png -o -2 -p 5 -c 16 -m ncc -rs 1 -f -ft gaussian -cs 3 -fl -flt wmf
python dp_stereo_matching.py -l ./source_images/tsukuba-l.png -r ./source_images/tsukuba-r.png -s ./dp_images/dp_tsukuba.png -o 35 -p 5 -c 16 -m grad -rs 1 -f -ft gaussian -cs 3 -fl -flt wmf

