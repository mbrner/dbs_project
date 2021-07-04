BASEDIR=$(dirname "$0")
echo $BASEDIR
if [ ! -f  $BASEDIR/WDIEXCEL.xlsx ]; then
    wget -nc https://databank.worldbank.org/data/download/WDI_excel.zip -O $BASEDIR/WDI_excel.zip
    unzip $BASEDIR/WDI_excel.zip -d $BASEDIR && rm $BASEDIR/WDI_excel.zip
else
    echo "File present!"
fi
