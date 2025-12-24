```python
from rsf.proj import *

datapath = '/media/lzf/Work/code/python/2025_work/direct_image_and_inversion_make_target/'


Flow ('hybird',[datapath+'dagang_deblending/testdata/d1.dat'],'bin2rsf n1=%d n2=%d| put n1=%d n2=%d n3=%d d1=0.001 d2=1 d3=1'%(n1,n2,n1,n2,n3))

Flow ('hybird1','hybird','put n1=%d n2=%d n3=%d d1=0.001 d2=1 d3=1'%(n1,n2,n3))

Flow ('hybird2','hybird1','window f1=0 n1=2000')

Flow ('hybird3','hybird2','bandpass flo=10 fhi=100')

Result ('hybird3',
            '''
            grey title= label1='Time' unit1='s' label2='Trace' unit2=''
            color=g screenratio=1.5 clip=%f wanttitle=n bar=y scalebar=n labelfat=4
            '''%(clip))


End ()
```