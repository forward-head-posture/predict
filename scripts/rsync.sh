REPO="$(cd $(dirname $(dirname $0)) && pwd -P)"
rsync -avzh --exclude-from=$REPO/.gitignore $REPO mp13:~/work/predict
