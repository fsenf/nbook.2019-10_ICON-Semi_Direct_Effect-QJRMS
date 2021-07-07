#!/bin/bash


# montage -tile 3x1 -trim -geometry +150+0 prof-cc.png  prof-qv.png  prof-temp.png average_profiles.png

convert west-east_qc.png  west-east_Fqc.png  west-east_Fw.png +append west-east.png
