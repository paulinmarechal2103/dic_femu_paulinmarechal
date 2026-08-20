//+
SetFactory("OpenCASCADE");

L = 10;
e = 0.25;

// --- ÉPROUVETTE TOURNÉE DE 90° (Inversion X/Y) ---

// Box initiale : Box(1) = {-L, -3, 0, 2*L, 6., e};
Box(1) = {-3, -L, 0, 6., 2*L, e};

// Cylindres initiaux (ex: {-L+3, -3, ...})
Cylinder(2) = {-3, -L+3, 0, 0, 0, 1, 2, 2*Pi};
Cylinder(3) = {-3, L-3, -0, 0, 0, 1, 2, 2*Pi};
Cylinder(4) = {3, -L+3, 0, 0, 0, 1, 2, 2*Pi};
Cylinder(5) = {3, L-3, -0, 0, 0, 1, 2, 2*Pi};

//+
BooleanDifference{ Volume{1}; Delete ; }{ Volume{2}; Volume{3};  Volume{4}; Volume{5}; Delete ; }

// Découpe supérieure initiale : Box(2) = {-L+3, 1., 0, 2*(L-3.), 10., e};
Box(2) = {1., -L+3, 0, 10., 2*(L-3.), e};
BooleanDifference{ Volume{1}; Delete ; }{ Volume{2}; Delete ; }

// Découpe inférieure initiale : Box(2) = {-L+3, -11., 0, 2*(L-3.), 10., e};
Box(2) = {-11., -L+3, 0, 10., 2*(L-3.), e};
BooleanDifference{ Volume{1}; Delete ; }{ Volume{2}; Delete ; }

// Zone centrale initiale : Box(2) = {-L+3,-5.3,-2,2*(L-3.),10,4}; 
Box(2) = {-5.3, -L+3, -2, 10, 2*(L-3.), 4}; 

v() = BooleanIntersection{Volume{2} ; Delete;}{ Volume{1}; } ;
w() = BooleanFragments{ Volume{v()} ; Delete; }{ Volume{1}; Delete ;} ; 

Characteristic Length{ PointsOf{ Volume{w()}; } } = 0.5;
Characteristic Length{ PointsOf{ Volume{v()}; } } = 0.5;
