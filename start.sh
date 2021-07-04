BASEDIR=$(dirname "$0")
echo $BASEDIR
$BASEDIR/data/get_wdi_data.sh
docker-compose up $1