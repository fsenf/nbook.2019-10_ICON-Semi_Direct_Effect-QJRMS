#!/bin/bash


# montage -tile 3x1 -trim -geometry +150+0 prof-cc.png  prof-qv.png  prof-temp.png average_profiles.png

convert prof-temp.png prof-qv.png prof-cc.png +append average_profiles.png
