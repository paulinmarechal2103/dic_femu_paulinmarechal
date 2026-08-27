// Paramètres de maillage
lc = 0.4;

// --- Paramètres géométriques globaux ---
H = 10.0;  // Longueur totale selon Y
W = 3;   // Largeur totale selon X
E = 0.25;  // Épaisseur selon Z

// --- Paramètres du profil de l'entaille (Spline) ---
X_fond = 0.8;      // Position X au fond des lobes (creux maximal)
X_centre_pointe = 0.9; // Position X de la pointe centrale (Y=0)
Y_lobe = 0.75;      // Position Y du fond des lobes
ent = 1; // entaille

// --- Points de la face arrière (Z = 0) ---

// Coins extérieurs de la plaque
Point(1) = {-W/2,  H/2, 0, lc};
Point(2) = {-W/2,  ent, 0, lc}; // Début entaille gauche
Point(3) = {-W/2, -ent, 0, lc}; // Fin entaille gauche
Point(4) = {-W/2, -H/2, 0, lc};

Point(5) = {W/2, -H/2, 0, lc};
Point(6) = {W/2, -ent, 0, lc};  // Début entaille droite
Point(7) = {W/2,  ent, 0, lc};  // Fin entaille droite
Point(8) = {W/2,  H/2, 0, lc};

// Points de contrôle pour la Spline GAUCHE (X négatif)
Point(9)  = {-X_fond,   Y_lobe, 0, lc}; // Fond du lobe supérieur gauche
Point(10) = {-X_centre_pointe,  0, 0, lc}; // Pointe centrale gauche
Point(11) = {-X_fond,  -Y_lobe, 0, lc}; // Fond du lobe inférieur gauche

// Points de contrôle pour la Spline DROITE (X positif)
Point(12) = {X_fond,   -Y_lobe, 0, lc}; // Fond du lobe inférieur droit
Point(13) = {X_centre_pointe,   0, 0, lc}; // Pointe centrale droite
Point(14) = {X_fond,    Y_lobe, 0, lc}; // Fond du lobe supérieur droit


// --- Lignes du contour ---
Line(1) = {1, 2};

// Entaille gauche interpolée par UNE SEULE spline fluide
Spline(2) = {2, 9, 10, 11, 3};

Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};

// Entaille droite interpolée par une spline
Spline(6) = {6, 12, 13, 14, 7};

Line(7) = {7, 8};
Line(8) = {8, 1};


// --- Création de la surface plane ---
Curve Loop(1) = {1, 2, 3, 4, 5, 6, 7, 8};
Plane Surface(1) = {1};

// --- Extrusion selon Z ---
Extrude {0, 0, E} {
  Surface{1};
}

// Volume physique pour le maillage 3D
Physical Volume("Eprouvette_Spline") = {1};
