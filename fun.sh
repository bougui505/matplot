#!/usr/bin/env zsh
cat << EOF

$(date): sourcing $0
EOF

MYTMP=$(mktemp -d)  # Temporary directory for the current script. Use it to put temporary files.
trap '/bin/rm -rf "$MYTMP"' EXIT INT  # Will be removed at the end of the script

precmd() {
    [ -f fun.sh ] && source fun.sh
}

test_plot2_simple () {
    seq 10 |shuf | plot2
}

test_plot2_scatter () {
    paste -d, =(seq 100|shuf) =(seq 100|shuf) =(seq 200 500|shuf) =(seq 200 500|shuf) | plot2 --fields x y x y --scatter -d,
}

test_plot2_moving_average () {
    paste -d, =(seq 1000) =(seq 1000|shuf) =(seq 500) =(seq 500|shuf) | plot2 -d, --fields x y x y --mov 10
}

test_plot2_pca () {
    paste -d, =(seq 100|shuf) =(seq 100|shuf) =(seq 200 500|shuf) =(seq 200 500|shuf) | plot2 --fields x y x y -d, --pca
}

test_plot2_pca_z () {
    paste -d, =(seq 1000|shuf) =(seq 1000|shuf) | awk -F"," 'BEGIN{OFS=","}{print $1,$2,NR<500}' | plot2 -d, --fields x y z --pca --cmap jet
}

test_plot2_overlap () {
    cat \
        =(np --nopipe 'A=np.random.normal(loc=0, scale=1, size=(1000,2));print_(A)' | awk '{print $0,0}') \
        =(np --nopipe 'A=np.random.normal(loc=(2,2), scale=1, size=(1000,2));print_(A)' | awk '{print $0,1}') \
        =(np --nopipe 'A=np.random.normal(loc=(8,-2), scale=1, size=(1000,2));print_(A)' | awk '{print $0,2}') \
        | plot2 --pca --fields x y z &
    cat \
        =(np --nopipe 'A=np.random.normal(loc=0, scale=1, size=(1000,2));print_(A)' | awk '{print $0,0}') \
        =(np --nopipe 'A=np.random.normal(loc=(2,2), scale=1, size=(1000,2));print_(A)' | awk '{print $0,1}') \
        =(np --nopipe 'A=np.random.normal(loc=(8,-2), scale=1, size=(1000,2));print_(A)' | awk '{print $0,2}') \
        | plot2 --no_over --fields x y z
}
