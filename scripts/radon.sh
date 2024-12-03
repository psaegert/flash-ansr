# If the results/radon directory does not exist, create it
if [ ! -d "results/radon" ]; then
  mkdir results/radon
fi

radon cc src/ -a -j > results/radon/cc.json
radon mi src/ -j  > results/radon/mi.json
radon raw src/ -j  -s > results/radon/raw.json
radon hal src/ -j  > results/radon/hal.json