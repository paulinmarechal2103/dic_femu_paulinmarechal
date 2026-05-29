//+
SetFactory("OpenCASCADE");

L =10;
e = 0.25;

Box(1) = {-L, -3, 0, 2*L, 6., e};

//+
Cylinder(2) = {-L+3, -3, 0, 0, 0, 1, 2, 2*Pi};
Cylinder(3) = {L-3, -3, -0, 0, 0, 1, 2, 2*Pi};
Cylinder(4) = {-L+3, 3, 0, 0, 0, 1, 2, 2*Pi};
Cylinder(5) = {L-3, 3, -0, 0, 0, 1, 2, 2*Pi};

//+
BooleanDifference{ Volume{1}; Delete ; }{ Volume{2}; Volume{3};  Volume{4}; Volume{5}; Delete ; }

Box(2) = {-L+3, 1., 0, 2*(L-3.), 10., e};
BooleanDifference{ Volume{1}; Delete ; }{ Volume{2}; Delete ; }

Box(2) = {-L+3, -11., 0, 2*(L-3.), 10., e};
BooleanDifference{ Volume{1}; Delete ; }{ Volume{2}; Delete ; }

//Box(1) = {-1,-1,-1,2,2,2}; 
Box(2) = {-L+3,-5.3,-2,2*(L-3.),10,4}; 
v() = BooleanIntersection{Volume{2} ; Delete;}{ Volume{1}; } ;
//NumberSpheresIn = #v[] ;  
//Physical Volume(1) = { v() } ; 
w() = BooleanFragments{ Volume{v()} ; Delete; }{ Volume{1}; Delete ;} ; 



Characteristic Length{ PointsOf{ Volume{w()}; } } = 0.5;
Characteristic Length{ PointsOf{ Volume{v()}; } } = 0.5;
