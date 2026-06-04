// Fichier .geo pour GMSH
// Carré de 10x10 avec une épaisseur de 0.25 et 3 trous cylindriques non superposés
// Utilisation de l'extrusion 3D à partir d'une surface 2D percée

SetFactory("OpenCASCADE"); // Recommandé pour les géométries complexes et les trous

lc = 0.5;

// ---- PLAQUE DE BASE (Carré 10x10) ----
Point(1) = {0,  0,  0, lc};
Point(2) = {10, 0,  0, lc};
Point(3) = {10, 10, 0, lc};
Point(4) = {0,  10, 0, lc};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Curve Loop(1) = {1, 2, 3, 4};

// ---- TROU 1 : Rayon 2.0 ----
cx1 = 5.41; cy1 = 3.47; r1 = 2.0;
p1 = newp; Point(p1) = {cx1, cy1, 0, lc};
p2 = newp; Point(p2) = {cx1 + r1, cy1, 0, lc};
p3 = newp; Point(p3) = {cx1, cy1 + r1, 0, lc};
p4 = newp; Point(p4) = {cx1 - r1, cy1, 0, lc};
p5 = newp; Point(p5) = {cx1, cy1 - r1, 0, lc};

c1 = newc; Circle(c1) = {p2, p1, p3};
c2 = newc; Circle(c2) = {p3, p1, p4};
c3 = newc; Circle(c3) = {p4, p1, p5};
c4 = newc; Circle(c4) = {p5, p1, p2};

Curve Loop(2) = {c1, c2, c3, c4};

// ---- TROU 2 : Rayon 1.0 ----
cx2 = 8.26; cy2 = 7.97; r2 = 1.0;
p6 = newp; Point(p6) = {cx2, cy2, 0, lc};
p7 = newp; Point(p7) = {cx2 + r2, cy2, 0, lc};
p8 = newp; Point(p8) = {cx2, cy2 + r2, 0, lc};
p9 = newp; Point(p9) = {cx2 - r2, cy2, 0, lc};
p10 = newp; Point(p10) = {cx2, cy2 - r2, 0, lc};

c5 = newc; Circle(c5) = {p7, p6, p8};
c6 = newc; Circle(c6) = {p8, p6, p9};
c7 = newc; Circle(c7) = {p9, p6, p10};
c8 = newc; Circle(c8) = {p10, p6, p7};

Curve Loop(3) = {c5, c6, c7, c8};

// ---- TROU 3 : Rayon 0.5 ----
cx3 = 4.74; cy3 = 6.31; r3 = 0.5;
p11 = newp; Point(p11) = {cx3, cy3, 0, lc};
p12 = newp; Point(p12) = {cx3 + r3, cy3, 0, lc};
p13 = newp; Point(p13) = {cx3, cy3 + r3, 0, lc};
p14 = newp; Point(p14) = {cx3 - r3, cy3, 0, lc};
p15 = newp; Point(p15) = {cx3, cy3 - r3, 0, lc};

c9 = newc; Circle(c9) = {p12, p11, p13};
c10 = newc; Circle(c10) = {p13, p11, p14};
c11 = newc; Circle(c11) = {p14, p11, p15};
c12 = newc; Circle(c12) = {p15, p11, p12};

Curve Loop(4) = {c9, c10, c11, c12};

// ---- SURFACE FINALE PERCÉE ----
// La surface Plane prend le contour extérieur (1) et exclut les contours intérieurs (2, 3, 4)
Plane Surface(1) = {1, 2, 3, 4};

// ---- EXTRUSION POUR CRÉER LE VOLUME 3D ----
// L'extrusion de cette surface crée automatiquement le bloc 3D avec les trous traversants
Extrude {0, 0, 0.25} {
  Surface{1};
}


Characteristic Length{ PointsOf{ Volume{w()}; } } = 0.5;
Characteristic Length{ PointsOf{ Volume{v()}; } } = 0.5;

