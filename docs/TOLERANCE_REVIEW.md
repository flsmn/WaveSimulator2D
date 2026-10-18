# Prüfung der physikalischen Grenzwerte – 18. Oktober 2026

Geprüfter Stand: Engine 0.1.0.dev4 / `10f7ce9`, Studio 0.1.0.dev8. Grundlage: ursprüngliches Protokoll und Grenzwerte aus `d06d04b`, tatsächlicher Prüfcode, erneut ausgeführte dev4-Referenzen sowie vorhandene Mac-Qualitätsberichte. Rohdaten: [tolerance-review-results.json](tolerance-review-results.json), [dev4-Referenzlauf](physical-results-dev4-review-input.json).

**Bewertung: Die 15-/20-%-Grenzen reichen als grobe Demonstrationsprüfungen, nicht als quantitative Genauigkeitsabnahme.** Ihr Einfrieren vor der Messung verhindert nachträgliches Anpassen an Ergebnisse, begründet aber nicht ihre fachliche Höhe. Die vorhandenen Erläuterungen nennen plausible Fehlerquellen, quantifizieren deren Beiträge nicht. PRD/Development Plan verlangen begründete Toleranzen, schreiben diese Prozentzahlen jedoch nicht vor. Frühere PASS-Berichte bleiben unverändert als historische v1-Prototypnachweise erhalten. Numerischer Solver, Qualitätsauswahl und aktive Grenzwerte wurden bei dieser Prüfung nicht geändert.

## Was die hohen Grenzen tatsächlich erlauben

- T01 erlaubt 15 % Unterschied der Feldamplitude zwischen nur zwei Rastern an einem Beobachtungspunkt. Das ist kein Nachweis eines absoluten Amplitudenfehlers von höchstens 15 %. Schon +15 % Amplitude entsprechen +32,25 % Intensität bei I proportional zu u². Eine gemeinsame systematische Abweichung beider Raster würde hier unentdeckt bleiben.
- T03 erlaubt 15 % Differenz zur Zweipunkt-Referenz des Doppelspalts. Die Simulation enthält jedoch endlich breite/dicke Öffnungen und Punktquellenbeleuchtung; die Referenz behandelt die beiden Öffnungen als ideale Punkte. Die Unsicherheit dieser Näherung wurde nicht bestimmt. Der Bericht mischt damit Diskretisierung, Geometrie-/Referenznäherung und Auswertung.
- T04 erlaubt 20 % Differenz zwischen einem wellenoptischen axialen Intensitätsmaximum und einer paraxialen geometrischen Schnittweite. Bei der Referenz 23,53358 sind das ±4,70672 Referenzlängen. Die groben Prüfraster haben nur 5,33 bzw. 6,67 Zellen pro kürzester Materialwellenlänge und erfüllen damit nicht einmal die aktuelle 12-Zellen-Warnschwelle. Ihre Fokusabweichungen von 18,5/11,5 % tragen keine quantitative Freigabe.
- Die zusätzlich erlaubten 10 % Fokusverschiebung zwischen zwei Rastern sind lediglich eine Rasterempfindlichkeitsgrenze. Zwei ähnliche Ergebnisse können denselben Fehler haben. Sie liefern weder die beobachtete Konvergenzordnung noch eine belastbare Restfehlerschranke.
- Eine Fokusverstärkung ≥1,5 zeigt einen Fokussierungseffekt, prüft aber weder genaue Fokuslage noch korrekte Spitzenintensität oder Fokusbreite.

## Konkrete Schwächen der Messung

1. Die Fokusposition wird aus 321 Punkten über 0,8 Referenzbrennweiten als diskretes argmax gewählt. Ein Abstand beträgt 0,25 % der Referenz bzw. 0,05883 Längeneinheiten. Die früher gemeldete feine Fokusverschiebung von 0,25 % ist genau **ein Abtastschritt**. Sie ist kein Nachweis einer Genauigkeit von 0,25 %. Es fehlt eine Unsicherheit der Peak-Lokalisierung.
2. Der Doppelspalt wird in Schritten von 0,0625 ausgewertet. Allein eine halbe Abtastzelle entspricht 0,833 % der idealen Minimumposition. Ein engerer Positionsnachweis braucht eine feinere Auswertung oder einen geprüften lokalen Fit mit Unsicherheitsangabe; eine Interpolation erzeugt dabei keine zusätzliche Auflösung des numerischen Feldes.
3. Der getriebene T01-Wellenlängenfehler steigt beim feineren Raster von 0,606 auf 0,861 %. Das ist kein monotones Konvergenzergebnis. Der Messansatz für eine zylindrische Punktquellenwelle, Restreflexion und das endliche Messfenster müssen mitgeprüft werden.
4. Raum- und Zeitschritt werden zusammen verändert. Separate Zeitschrittempfindlichkeit sowie Einfluss von Einlaufzeit, Messfenster, Domänengröße/Randzone und Konturdiskretisierung sind nicht zu einer Fehlerbilanz zusammengeführt.
5. Die Qualitätsprüfungen in Studio enthalten erneut fest codierte 0,15/0,20-Grenzen; auch die Einfallswinkelgrenze im Engine-Prüfskript ist direkt auf 3° gesetzt. Eine zukünftige Revision muss eine gemeinsame versionierte Parameterquelle verwenden.
6. Die Phasendriftbudgets Preview π, Balanced π/2 und High Quality π/4 sind heuristische Bild-/Prototypbudgets. Sie belegen keine genaue kohärente Interferenz. Beispielsweise akkumuliert ein kleiner relativer Geschwindigkeitsfehler näherungsweise zu |Δφ| ≈ 2π N |ε| über N Wellenlängen: 2 % über 64 Wellenlängen ergeben etwa 8,04 rad. Eine beispielhafte Phasenforderung von 0,05 rad würde dort ungefähr 0,0124 % verlangen. Das ist keine neue globale Produktvorgabe, zeigt aber, warum ein lokales Geschwindigkeitslimit allein nicht genügt.

## Empfohlene Ziele für eine neue quantitative Abnahme

Die folgenden Werte sind **vorgeschlagene technische Produktziele**, keine Normwerte, keine nachträglich aktivierte v2-Abnahme und keine bereits belegte allgemeine Genauigkeit. Sie sind vor einem neuen Abnahmelauf als neue Policy zu versionieren. Die alten Messwerte zeigen lediglich, wo diese Zielzahlen auf dem bisherigen Messverfahren bereits eingehalten werden.

| Messgröße | Bisher | Vorgeschlagenes Ziel | Bewertung vorhandener Daten |
| --- | --- | --- | --- |
| Geschwindigkeit in analytischen Kalibrierfällen | 2 % | 0,5 % | 0,079–0,405 %: erfüllt diese Zahl |
| Wellenlänge in kontrolliertem Quellenfall | 5 % | 1 % | 0,606–0,861 %: erfüllt Zahl; Messbias offen |
| Quellenamplitude | 15 % Zweirasterdifferenz | 2 % gegen geeignete absolute Referenz, zusätzlich Rasterprüfung | Absolute Referenz fehlt; bisher Punkt 0,448 %, Linie 2,880 % Rasterdifferenz |
| Brechungswinkel | 3° | 1° | 0,646°: erfüllt Zahl; endliche Beleuchtung separat prüfen |
| Homogene Dämpfungsrate | 5 % | 0,1 % | maximal 0,000383 %: erfüllt Zahl |
| Randreflexion, Amplitudenverhältnis | 8 % | 1 % im festgelegten Bereich | 0,482/0,904 %: nur der getestete 8-Einheiten-Rand erfüllt Zahl |
| Doppelspalt-Minimum | 15 % gegen Zweipunktnäherung | 2 % gegen passende endliche Wellenreferenz | Neue Referenz fehlt; alte Abweichung 3,39–6,72 % |
| Symmetrie im festgelegten symmetrischen Aufbau | 3 % | 0,1 % | maximal 0,0000212 %: erfüllt Zahl |
| Linsen-Fokusposition | 20 % gegen paraxiale Formel | 2 % gegen unabhängige passende skalare Wellenreferenz | Neue Referenz fehlt; bisher nicht quantitativ abgenommen |
| Numerische Unsicherheit der Fokuslage | 10 % Zweirasterverschiebung | ≤1 % aus belastbarer Konvergenz-/Fehleranalyse | Nicht nachgewiesen |

Das Nullfeld auf den Wänden bleibt als fester Invariantentest erhalten. Gain≥1,5 und die paraxiale Brennweite bleiben als qualitative Plausibilitätsvergleiche nutzbar, erhalten aber keine Bedeutung als genaue Wellenreferenz. Die feinen MPS-Linsenwerte 6,00/6,25 % zur paraxialen Formel beweisen weder einen gleich großen numerischen Fehler noch dessen Abwesenheit.

Ein hypothetisches Ersetzen der bisherigen Grenzzahl durch 2 % lässt auf derselben alten Auswertung Linienquellen-Rasterdifferenz, Doppelspaltposition und Linsenposition durchfallen. Das allein diagnostiziert keinen Solverfehler: insbesondere für Spalt und Linse muss zuerst die Referenz zur simulierten Messgröße passen. Die maschinenlesbare Gegenprüfung ist entsprechend als kontrafaktische Auswertung gekennzeichnet.

## Erforderlicher Nachweis

- Einen analytischen Referenzfall mit klaren Rand-/Quellenbedingungen für homogene Ausbreitung und Quellennormierung verwenden; für Material-/Linsenfälle eine unabhängige Lösung derselben skalaren Gleichung mit gleicher Geometrie, Beleuchtung und Messdefinition heranziehen. Eine feinere Rechnung derselben Engine ist eine Selbstkonvergenzprüfung, kein unabhängiger physikalischer Nachweis.
- Für die bisher unklaren Spalt-/Linsenergebnisse eine dritte gezielte Rasterstufe ergänzen. Zeit-, Rand-/Domänen-, Geometrie- und Messfensterfehler gesondert eingrenzen. Die bisherige Vereinbarung, eine dritte Stufe bei unklaren Abweichungen zu ergänzen, greift hier.
- Die beobachtete Konvergenzordnung/den asymptotischen Bereich prüfen. Richardson/GCI nur bei erfüllten Voraussetzungen verwenden; insbesondere an treppenförmigen Interfaces oder bei nichtmonotonen Extremwerten nicht pauschal Ordnung zwei annehmen. Bei fehlendem Nachweis bleibt die Unsicherheit offen.
- Fokus-/Minimumpunkte mit einer kontrollierten Auswertungsauflösung und Unsicherheitsangabe bestimmen. Ganze Profile, Fokusbreite und relative Intensität zusätzlich prüfen; die Phasenanforderung aus der beabsichtigten Interferenzmessung ableiten.
- Erst nach diesem Nachweis die tatsächlich geprüften Qualitätsstufen quantitativ freigeben. Balanced-/High-Quality-Linsenfokus bleibt offen. Hohe Rechenkosten oder ein Ressourcen-Pass rechtfertigen kein weiteres Genauigkeitsversprechen.

## Fachliche Quellen und ihre Grenzen

[NASA: räumliche Gitterkonvergenz](https://www.grc.nasa.gov/www/wind/valid/tutorial/spatconv.html) empfiehlt drei Raster für beobachtete Ordnung und Prüfung des asymptotischen Bereichs. Das Verfahren liefert unter Voraussetzungen eine numerische Fehlerschätzung; eine kleine Zweirasterdifferenz allein genügt nicht. Die Quelle setzt keinen für unsere Software gültigen Prozentgrenzwert fest.

[NASA: Fehlerarten](https://www.grc.nasa.gov/www/wind/valid/tutorial/errors.html) unterscheidet Modell-/Geometriefehler von räumlichen und zeitlichen Diskretisierungsfehlern. Diese Trennung fehlt in der bisherigen pauschalen Spalt-/Linsentoleranz.

[Li, JOSA 72, 770–774 (1982)](https://doi.org/10.1364/JOSA.72.000770) untersucht beugungsbedingte Fokusverschiebungen bei endlicher kreisförmiger Apertur. Das stützt die Notwendigkeit einer wellenoptischen Referenz, liefert aber weder einen 20-%-Freibrief noch eine quantitative Korrektur für unsere zweidimensionale Linse.

## Ergänzung zur Winkelreferenz und zum Alt/Neu-Vergleich

Die tatsächlichen Fehlerformeln und ihr Nenner sind in [ERROR_METRICS.md](ERROR_METRICS.md) erläutert. T02 verwendet den vorgegebenen Einfallswinkel 25° für Snell. Mit dem tatsächlich gemessenen Einfallswinkel wären es 1,037592° statt 0,645968° Brechungsabweichung. Die vorherige Aussage, dass die gespeicherte Metrik die vorgeschlagene 1°-Zahl einhält, bleibt auf genau diese v1-Metrik begrenzt; eine strengere bedingte Brechungsprüfung ist damit nicht bestanden. Die zusätzliche Diagnose ändert keine v1-Rohdaten oder aktive Grenzwerte. Ein quellenfreier Alt/Neu-Vergleich mit analytisch angeglichener Zeitskala ist ebenfalls dokumentiert; er ist keine physikalische Gesamtfreigabe.
