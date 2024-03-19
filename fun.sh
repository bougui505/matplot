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
    seq 10 |shuf | plot2 --title "Simple plot"
}

test_plot2_hist () {
    np --nopipe "
a=np.random.normal(loc=0, size=100)
b=np.random.normal(loc=1, size=100)
out = np.c_[a, b]
print_(out)
" \
    | plot2 --title "Histogram" -H --fields y y --labels h1 h2 --alpha 0.5
}

test_plot2_ylim () {
    seq 10 |shuf | plot2 --title "Simple plot" --ymin 5 --ymax 7
}

test_plot2_xlim () {
    seq 10 |shuf | plot2 --title "Simple plot" --xmin 0.5 --xmax 0.7
}

test_plot2_semilogy () {
    seq 10 |shuf | plot2 --title "Simple plot" --semilog y
}

test_plot2_semilogx () {
    seq 10 |shuf | plot2 --title "Simple plot" --semilog x
}

test_plot2_semilogxy () {
    seq 10 |shuf | plot2 --title "Simple plot" --semilog x y
}

test_plot2_scatter () {
    paste -d, =(seq 100|shuf) =(seq 100|shuf) =(seq 200 500|shuf) =(seq 200 500|shuf) | plot2 --fields x y x y --scatter -d,
}

test_plot2_scatter_markers () {
    paste =(seq 100|shuf) =(seq 100|shuf) =(seq 100) \
        | awk '{if (NR<50){print $0,"^"}else{print $0,"r*"}}' \
        | plot2 --fields x y z m --scatter
}

test_plot2_moving_average () {
    paste -d, =(seq 1000) =(seq 1000|shuf) =(seq 500) =(seq 500|shuf) | plot2 -d, --fields x y x y --mov 10
}

test_plot2_pca () {
    paste -d, =(seq 100|shuf) =(seq 100|shuf) =(seq 200 500|shuf) =(seq 200 500|shuf) | plot2 --fields x y x y -d, --pca --aspect 5 5
}

test_plot2_pca_gaussian () {
    np --nopipe 'A=np.random.normal(loc=0, scale=(1,2)*np.asarray([3, 1]), size=(1000,2));print_(A)' \
        | plot2 --fields x y --pca --orthonormal
}

test_plot2_pca_markers () {
    paste =(seq 100|shuf) =(seq 100|shuf) \
        | awk '{if (NR<50){print $0,"0","^"}else{print $0,"1","r*"}}' \
        | plot2 --fields x y z m --pca
}

test_plot2_pca_z () {
    paste -d, =(seq 1000|shuf) =(seq 1000|shuf) | awk -F"," 'BEGIN{OFS=","}{print $1,$2,NR<500}' | plot2 -d, --fields x y z --pca --cmap jet
}

test_plot2_overlap () {
    # cat \
    #     =(np --nopipe 'A=np.random.normal(loc=0, scale=1, size=(1000,2));print_(A)' | awk '{print $0,0}') \
    #     =(np --nopipe 'A=np.random.normal(loc=(2,2), scale=1, size=(1000,2));print_(A)' | awk '{print $0,1}') \
    #     =(np --nopipe 'A=np.random.normal(loc=(8,-2), scale=1, size=(1000,2));print_(A)' | awk '{print $0,2}') \
    #     | plot2 --pca --fields x y z &
    cat \
        =(np --nopipe 'A=np.random.normal(loc=0, scale=1, size=(1000,2));print_(A)' | awk '{print $0,0}') \
        =(np --nopipe 'A=np.random.normal(loc=(2,2), scale=2, size=(1000,2));print_(A)' | awk '{print $0,1}') \
        =(np --nopipe 'A=np.random.normal(loc=(8,-2), scale=0.5, size=(1000,2));print_(A)' | awk '{print $0,2}') \
        =(np --nopipe 'A=np.random.normal(loc=(8,-1), scale=2, size=(1000,2));print_(A)' | awk '{print $0,3}') \
        =(np --nopipe 'A=np.random.normal(loc=(6,5), scale=1, size=(1000,2));print_(A)' | awk '{print $0,4}') \
        | plot2 --no_over --fields x y z --orthonormal
}

test_plot2_repulsion () {
    cat \
    =(np --nopipe 'A=np.random.normal(loc=0, scale=1, size=(1000,2));print_(A)' | awk '{print $0,0}') \
    =(np --nopipe 'A=np.random.normal(loc=(5,5), scale=1, size=(1000,2));print_(A)' | awk '{print $0,1}') \
    | plot2 --fields x y z --orthonormal --scatter --repulsion 0.5
}

test_plot2_save_read () {
    seq 10 | plot2 --save test.png
    plot2 --read test.png
}

test_plot2_text () {
    cat << EOF | plot2 --scatter --fields x y z m t --cmap tab10
0 0 0 o -
1 2 1 o -
0 1 0 r* 0
1 3 1 r* 1
EOF
}

test_plot2_graph () {
    paste =(seq 10|shuf) =(seq 10|shuf) =(seq 10 | awk '{if ($1<5){print 0}else{print 1}}')\
        | awk '{print $1,$2,$3,$3}' \
        | plot2 --graph --fields x y z ed
}

test_plot2_linear () {
    seq 100 | awk '{print $1/10+2*rand(),$1/10+2*rand()}' \
        | plot2 --fields x y --scatter --pcr
}

test_plot2_sigma (){
    # Testing plotting standard deviation
    seq 100 \
        | awk '{
            n=100
            s=0
            s2=0
            for (i=1;i<=n;i++){
                x=rand()
                s+=x
                s2+=x**2
            }
            mu=s/n
            mu2=s2/n
            sigma=mu2-mu**2
            print(mu,sigma)
    }' \
        | plot2 --fields y e
}

test_plot2_boxplot (){
    paste -d"," =(seq 50) =(seq 100) =(seq 80) \
        | plot2 -d"," --boxplot --fields y y y

}

test_plot2_boxplot_xy (){
    paste -d"," =(seq 100) =(seq 100) \
        | plot2 -d"," --boxplot --fields x y --bins 10

}
