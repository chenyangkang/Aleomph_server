for city in $(<all_city_code.txt)
do
{
    echo ${city}
    echo '''
    cd /beegfs/store4/chenyangkang/06.ebird_data/42.City_and_migration/02.pymc
    city='"${city}"'
    python model_level_3_May18.py ${city}

    wait


    ''' > ${city}.sh

    chmod 755 ${city}.sh

    while true
    do
        {
            myjobcount=`qstat |grep "chenyangkang"|wc -l`
            if [[ $myjobcount -lt 20 ]];then
                qsub -q cu -l nodes=1:ppn=4,mem=20G ${city}.sh
                break
            fi
            sleep 5
        }
    done
    rm ${city}.sh
    sleep 0.5

    }
done
