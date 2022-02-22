@echo off

set fusiondir=%1
set points=%2
set output=%3

%fusiondir%GroundFilter /finalsmooth /median:3 /iterations:20 %output%ground_points.las 3.5 %points%
%fusiondir%GridSurfaceCreate %output%dtm.dtm 0.5 m m 0 0 0 0 %output%ground_points.las
%fusiondir%DTM2TIF %output%dtm.dtm %output%dtm_refined.tif