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
