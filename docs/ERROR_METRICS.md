# Fehlermaße und Vergleich mit der Ausgangsengine

Stand: 18. Oktober 2026, Engine 0.1.0.dev4. Diese Erläuterung beschreibt den tatsächlichen Prüfcode in [run_references.py](../validation/run_references.py), nicht einen allgemeinen Gesamtfehler der Engine. Die Grenzen/Referenzschwächen stehen in [TOLERANCE_REVIEW.md](../validation/TOLERANCE_REVIEW.md). Keine aktive Toleranz oder Solvergleichung wurde hier geändert.

## Referenzabweichung

Für eine von null verschiedene Referenzgröße q_ref wird `e = abs(q_sim/q_ref - 1)` berechnet. Die JSON-Zahl ist ein Bruch; für Prozent wird mit 100 multipliziert. Beispiel: Geschwindigkeit 0,996831 statt 1 ergibt 0,003169 bzw. 0,3169 %. Ein Fehler wird nur für die jeweils gemessene Größe angegeben, nicht für das gesamte Wellenfeld.

- **Geschwindigkeit:** Ein bekanntes sinusförmiges Eigenfeld wird initialisiert und das fortgeschriebene Feld nach jedem Schritt darauf projiziert. Aus 180 Projektionswerten wird die zeitliche Frequenz der Rekursion geschätzt. Mit bekanntem räumlichem Wellenvektor gilt c_gemessen = omega_gemessen / |k|; Referenz c=1/n. Der Zeitnenner ist der numerische Zeitschritt, nicht GUI-Zeit/FPS.
- **Wellenlänge:** Die eingeschwungene Punktquellenantwort wird bei der Anregungsfrequenz komplex demoduliert. Aus der entfalteten Phase entlang der Beobachtungslinie wird eine räumliche Steigung k angepasst; lambda_gemessen = 2*pi/abs(k). Referenz hier lambda=4. Die nicht exakt plane Welle, Randreflexion und endliche Messzeit können diesen Schätzer beeinflussen.
- **Dämpfung:** Die Eigenmoden-Zeitreihe wird als Rekursion a[k+1] = A*a[k] + B*a[k-1] angepasst. Mit b=-B ergibt sich gamma_gemessen = 2*(1-b)/(dt*(1+b)). Referenz gamma=0,2. Auch hier bezeichnet der relative Fehler diese eine Messgröße.
- **Doppelspalt:** Die erste lokale Intensitätsminimum-Position wird als Abstand von der Symmetrieachse bestimmt. Referenz ist die Zweipunkt-Wegdifferenz mit r_oben-r_unten=lambda/2, hier 3,752254. Ein gemessener Abstand 3,625 ergibt 3,3914 % Referenzabweichung. Endliche Spaltbreite/-dicke sind in dieser Näherung nicht vollständig enthalten; daher ist diese Zahl kein isolierter Diskretisierungsfehler.
- **Linse:** Gemessen wird die Entfernung des axialen Intensitätsmaximums von der Linsenhinterfläche, nicht die globale x-Koordinate. Referenz ist die paraxiale Schnittweite 23,53358068. Der gemessene Wert 20,82721890 ergibt 11,5 %. Das ist eine Abweichung von einer Näherungsformel; daraus folgt kein bewiesener rein numerischer Fehler von 11,5 %.

## Andere Maße mit anderen Nennern

| Messung | Tatsächliche Berechnung | Bedeutung |
| --- | --- | --- |
| Quellenamplitude über zwei Raster | abs(A_grob/A_fein - 1) | Rasterdifferenz, keine absolute Amplitudenfehlergrenze |
| Fokusverschiebung über zwei Raster | abs(x_fokus,grob-x_fokus,fein) / f_ref | Rasterempfindlichkeit relativ zur paraxialen Schnittweite, keine Restfehlerschätzung |
| Brechungswinkel | abs(theta_gemessen-theta_Snell) | Absolute Abweichung in Grad, kein Prozentfehler |
| Spaltsymmetrie | max(abs(I(y)-I(2*y_mitte-y))) / max(I) | Maximale Spiegelabweichung, normiert am Profilmaximum |
| Randrücklauf | abs(U_kurzeDomaene-U_langeDomaene) / abs(U_langeDomaene) | Komplexe Amplitudendifferenz gegen eine längere Domäne; bedingte Referenz, nicht absolute Fehlerfreiheit |
| Wandfeld | max(abs(u[Wand])) | Absoluter Feldwert, kein Prozentfehler |

U bezeichnet hier den demodulierten komplexen Feldphasor. Die Referenznenner sind in den festen Prüffällen nicht null. Für Nullgrößen wird kein relativer Genauigkeitsanspruch aus diesen Formeln abgeleitet.

Der Winkeltest verwendet derzeit theta_Snell = asin(sin(25 Grad)/1,5), also den **vorgegebenen** Einfallswinkel. Er misst tatsächlich 25,622869 Grad einfallend und 15,718442 Grad gebrochen. Gegen die vorgegebene Referenz 16,364410 Grad sind das 0,645968 Grad. Berechnet man Snell stattdessen mit dem gemessenen Einfallswinkel, erhält man 16,756034 Grad und 1,037592 Grad Differenz. Das ist eine zusätzliche Diagnose, keine rückwirkend geänderte v1-Auswertung. Die Beleuchtungs-/Winkelschätzung muss vor einer engeren Abnahme mit berücksichtigt werden.

## Direkter Alt/Neu-Vergleich

Reproduzierbarer Aufruf aus dem Engine-Repository:

```sh
../WaveStudio/.venv/bin/python validation/compare_original.py
```

Das [Vergleichsskript](../validation/compare_original.py) lädt die tatsächliche historische PyTorch-Fortschreibung aus Git-Commit `40f6c78`. Der anschließende CUDA-Auswahlfix `5620234` änderte diese numerische Gleichung nicht. Ausgabe: [original-comparison-results.json](../validation/results/original-comparison-results.json), einschließlich Quellhashes und Laufzeitangaben. Der Test läuft auf CPU/float32 mit einem Torch-Thread. Er umgeht lediglich die historische fehlerhafte initial_field-Formprüfung durch direktes Setzen beider Anfangsfelder; die alte Fortschreibung wird nicht geändert. Konstante Materialkoeffizienten erfordern keine Szenen-Neurasterung.

1. **Behauptete Einheitenskala prüfen:** Vier sinusförmige Eigenmoden, n=1/1,5, nahezu axial/diagonal. Bei eigenem Default-Zeitschritt ergibt die alte Engine ca. 0,560 statt 1 im Vakuum, die aktuelle ca. 0,996–0,997. Gegen c_ref=1/n sind es rund 44 % versus 0,32–0,47 %. Dies ist ein Kalibrierungsbefund zur angegebenen Geschwindigkeitsskala, kein pauschaler 44-%-Feldfehler aller alten Simulationen.
2. **Nach analytischer Einheitenangleichung vergleichen:** Derselbe homogene quellenfreie Versuch nutzt bei dx=1 die bekannte Abbildung t_neu=sqrt(0,316)*t_alt. Im kontrollierten neuen Lauf wird deshalb dt_neu=0,56213877 verwendet; die normale Engine-Vorgabe bleibt unverändert bei dt=0,5*dx. Die alte Nullfortsetzung wird einem neuen festen Nullrand mit gleichem Innengebiet zugeordnet. Nach 180 Schritten liegen die relativen L2-Felddifferenzen zwischen 6,38e-7 und 1,27e-6, also 0,0000638–0,000127 %. Der maximale absolute Feldunterschied ist 1,85e-6 bei Anfangsamplitude 1.

Dabei ist `e_L2 = norm(u_neu-u_alt,2) / norm(u_alt,2)`. Es wird auf dem gleichen Innengebiet zur gleichen **kalibrierten physikalischen Zeit** verglichen, ohne nachträgliche optimale Phasenverschiebung oder Amplitudenanpassung. Im ersten Teil werden Geschwindigkeiten verglichen; aufgrund verschiedener dt werden dort keine Feldbilder nach gleicher Schrittzahl als identische physikalische Zeit ausgegeben.

Die kleinen Felddifferenzen zeigen, dass der homogene Fortschreibungsansatz nach Kalibrierung weitgehend derselbe ist. Sie sind weder ein Beleg absoluter physikalischer Genauigkeit noch ein Test von Quellen, Dämpfung, internen Wänden, Spalt oder Linse. Diese Teile haben absichtlich andere Verträge. Ein vollständiger Alt/Neu-T01–T04-Vergleich liegt nicht vor.

Vor dem PyTorch-Fork lag die CuPy-Engine (`16de78f`). Ihr Quellcode enthält bereits denselben unnormierten Kernel und dieselbe mathematische Grundfortschreibung. Der historische CuPy/CUDA-Code wurde auf diesem Mac **nicht ausgeführt**; der CPU-Test des PyTorch-Forks ersetzt keinen nativen CuPy/CUDA-Backendvergleich.

Der alte Solver ist keine unabhängige exakte Referenz. Übereinstimmung zweier verwandter Implementierungen und physikalische Richtigkeit müssen getrennt belegt werden. Die strengere quantitative Spalt-/Linsenabnahme bleibt offen.
