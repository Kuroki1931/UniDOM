#!/bin/bash

EACH_ENV_NUM=2

set -euC

                                                                                     
# for env in fold_cloth1 fold_cloth3 fold_tshirt shape_rope push_rope shape_rope_hard push_rope_hard unfold_cloth1 unfold_cloth3 pour_water pour_soup whip_rope
for env in unfold_cloth1 unfold_cloth3 pour_water pour_soup whip_rope

do
    echo "solving ${env}"
    seq 0 1 $((${EACH_ENV_NUM}-1)) | xargs -P ${EACH_ENV_NUM} -I{} bash solve_each_env.sh {} ${env}
done

# pkill -f daxbenc