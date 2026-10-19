# Quantitative Physik – Arbeitsstand vom 18. Oktober 2026

**MAC-01 bleibt offen.** Neue, unabhängige Referenzen und eng begrenzte Teilprotokolle sind implementiert. Eine quantitative Freigabe aller Qualitätsstufen, des Doppelspalts oder der Linse liegt weiterhin nicht vor. Numerischer Laufzeit-Solver, vorhandene v1-Grenzen und Presetregeln wurden nicht verändert. Die neuen Prüfungen verwenden CPU/float32 für die Engine, komplexe double-precision Referenzrechnungen und SciPy 1.18.1 als reine Prüfabhängigkeit.

## 1. Gemeinsame historische Grenzwerte und Winkelreferenz

Die unveränderten historischen Werte bleiben in [tolerances.json](../validation/tolerances.json). Das neue [Mac-Profil](../validation/mac-presets-v1.json) erbt sie und ergänzt nur die schon zuvor eingefrorene 5-%-Fokusverschiebungsgrenze aus `MAC_ACCEPTANCE_PLAN.md`. Studio `scripts/check_quality.py` liest dieses Profil statt eigener Kopien von 0,15/0,20/0,03/0,05. Das ist eine Konsolidierung der v1-Demonstrationsprüfung, keine engere Genauigkeitsfreigabe.

T02 wurde erneut ausgeführt: [Winkeldiagnose](physical-t02-conditioned-diagnostic.json). Die v1-Metrik bleibt 0,6459678° und besteht ihre alte 3°-Grenze. Gegen Snell mit dem **gemessenen** Einfallswinkel 25,6228687° ergibt sich 1,0375920°. Beide Definitionen stehen nebeneinander im Ergebnis; `quantitative_angle_accepted=false`. Endliche Beleuchtung und Winkelfit müssen weiterhin getrennt untersucht werden.

## 2. Unabhängige Quellenreferenz und vorab gespeicherte Teilprotokolle

Mit `u=Re(U exp(i omega t))` und `f=A sin(omega t+phase) delta` ergibt sich im unendlichen homogenen Referenzmedium n=1:

`U(r) = -A exp(i phase) H0^(2)(omega r)/4`.

Die endliche Linie integriert dieselbe Green-Funktion in kontinuierlicher Bogenlänge mit Gauss-Legendre-Quadratur. Diese Referenz verwendet weder Engine-Stencil noch Rasterisierer oder Quellengewichte. Normierung/Phase werden zusätzlich über den Fluss auf einem kleinen Kreis und die Linienintegration gegen unabhängige adaptive Quadratur geprüft. Eine Phasenverschiebung oder Amplitudenanpassung an die Engine findet nicht statt. Herleitung aus dem Modellvertrag und den [Hankel-Funktionen des NIST DLMF](https://dlmf.nist.gov/10.2); siehe auch das [Helmholtz-Beispiel der SciPy-Dokumentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.hankel1.html) mit der dortigen Vorzeichenkonvention.

[Protokoll 2](../validation/quantitative-v2.json) und [revidiertes Gebietsprotokoll 2.1](../validation/quantitative-v2-wide.json) wurden jeweils vor den zugehörigen Läufen gespeichert. Berichte enthalten Policy-/Runner-Hashes und tatsächliche Parameter. Die 2-%-Grenze ist ein begrenztes technisches Pilotziel aus der Grenzwertprüfung, keine Norm und kein allgemeines Produktversprechen. Ein 2-%-Amplitudenfehler kann bereits 4,04 % Intensitätsfehler bedeuten. Die 0,5-%-Grenze je gezielter Sensitivität reserviert einen Viertelanteil dieses Pilotziels; die Stichproben bilden keine mathematisch bewiesene Gesamtfehlerschranke.

Die initialen 64×64-Domänen mit Randbreite 8 scheitern: [Punkt](quantitative-source-point-initial.json), [Linie](quantitative-source-line-initial.json). Auch bei dx=0,0625 verbleiben 2,9770/6,6142 % maximale komplexe Abweichung. Ein [Gebietsprobe-Lauf](quantitative-source-point-domain-probe.json) ist ausdrücklich explorativ. Für Protokoll 2.1 werden Gebiet und Randbreite auf 96×96/16 erhöht; Quelle und Proben gemeinsam verschoben. Fehlergrenzen bleiben unverändert.

Protokoll 2.1: lambda=4, Phase=0,3, drei Raster dx=0,25/0,125/0,0625; Qualifikation nur bei dx=0,0625, dt=0,03125. Dies sind 64 Zellen je Wellenlänge und kurze feste Beobachtungswege. Zwei Fenster ab Simulationszeit 192 mit Dauer 32, halber Zeitschritt, vergrößertes Gebiet und breiterer Rand werden geprüft. Die Phasoren stammen aus einem Cosinus-/Sinus-/DC-Fit. Stationäre Ausgangswellen sind die analytische Referenz; reale Einschalttransienten und Randrückläufe können die endlichen Fenster beeinflussen. Die Fensterprüfung begrenzt diese Einflüsse empirisch, beweist keinen exakten eingeschwungenen Zustand.

| Quelle | Max. komplexer Fehler | Max. Amplitudenfehler | Änderung späteres Fenster | Änderung breiterer Rand | Teilprotokoll |
| --- | --- | --- | --- | --- | --- |
| Punkt | 0.6222 % | 0.1930 % | 0.2579 % | 0.1694 % | Bestanden im definierten Teilumfang |
| Linie | 0.7503 % | 0.2844 % | 0.9809 % | 0.6032 % | Nicht bestanden |

Rohdaten: [quantitative-sources-wide.json](quantitative-sources-wide.json). Der Punktfall besteht seine 2-%-/0,5-%-Kriterien an den fünf festen Proben. Die Linie verfehlt die 0,5-%-Grenze beim späteren Fenster und beim breiteren Rand trotz kleiner direkter Feldabweichung. `selected_scope_passed=false` und `mac01_complete=false` bleiben korrekt erhalten. Die vier Linienproben, andere Quellenwellenlängen, Materialien, längere Wege und die App-Presets erben keinen pauschalen Nachweis.

## 3. Unabhängige frequenzbasierte Linsenreferenz

[helmholtz_volume.py](../validation/helmholtz_volume.py) löst für ein kompaktes Materialgebiet die Lippmann-Schwinger-Gleichung `U - G * [k²(n²-1) U] = U_inc`. Der Kern ist die auslaufende freie Green-Funktion. Mittelpunkts-Nyström, integrierte singuläre Selbstzelle, Flächenanteile an der Geometrie und FFT/GMRES bilden einen von der zeitabhängigen Engine unabhängigen Rechenweg. Kontinuität von U und seiner Normalableitung entspricht demselben skalaren Modell. Die Referenz ist eine einfache niedrigordentliche Implementierung; sie übernimmt keine Hochordnungszusagen aus der Literatur. Mathematische Grundlage: [Ambikasaran et al., 2015](https://arxiv.org/abs/1505.07157), insbesondere die Volumenpotentialdarstellung. [SciPy GMRES](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.gmres.html) liefert die iterative Lösung; sein algebraisches Residuum ist keine physikalische Fehlerschranke.

Kalibrierung: Ein penetrabler Kreis unter ebener Beleuchtung besitzt eine unabhängige analytische Zylinderreihenlösung. Bei dx=0,0125 beträgt die komplexe relative L2-Abweichung 0.1667 % und der maximale absolute Fehler bei einheitlicher Einfallsamplitude 0.00624914. Das feinste Kreisraster besteht die vorab festgelegten Einzelgrenzen; FFT-Matrixwirkung und singuläre Zellintegration sind zusätzlich gegen direkte Verfahren geprüft.

Linsenfall: dieselbe feste 161-Punkte-pro-Seite-Polygonkontur, R=24, Apertur=16, n=1,5, kontinuierliche Linie (12,7)–(12,41), lambda=0,8. Die Referenz beschreibt freie Ausstrahlung ohne äußere Dämpfungszone. Sie ist deshalb noch kein abgeglichener Nachweis des endlichen Engine-Gebiets und Messfensters.

| Referenz-dx | Fokusabstand ab Hinterfläche | Änderung des komplexen Profils zum vorigen Raster (L2) |
| --- | --- | --- |
| 0.05 | 22.11583278 | — |
| 0.025 | 21.99474647 | 3.9079 % |
| 0.0125 | 21.96506864 | 0.9749 % |

Die letzte Fokusverschiebung beträgt 0.1261 % der paraxialen Bezugsstrecke und besteht das 0,25-%-Einzelkriterium. Die volle komplexe Profiländerung von 0.9749 % liegt jedoch über der eingefrorenen 0,5-%-Grenze. Geometrie-Unterteilung, Quellenquadratur und Peak-Fit bestehen ihre Sensitivitätsprüfungen. **Die Referenz-Gesamtqualifikation bleibt nicht bestanden.** Aus einer kleinen Fokusverschiebung wird keine gesicherte Feld-/Intensitätsgenauigkeit abgeleitet.

Rohdaten: [vollständige Nachprüfung](volume-reference-followup.json). Erhaltene Vorläufer: [erster Teilbericht](volume-reference-initial.json), [Zellenbudget-Ablehnung](volume-reference-fine.json), [Iterationsbudget-Fehler](volume-reference-fine-budget.json). Der erste Bericht bleibt unvollständig nach einem behobenen NumPy-bool-Serialisierungsfehler. Die Zellenzulassung wurde für die tatsächlichen 281600 Zellen explizit auf 400000 korrigiert; beim GMRES-Folgelauf wurde nur die maximale Zahl der Restart-Zyklen 40→80 erhöht, das Residuumziel 1e-9 blieb unverändert. Bereits abgeschlossene Raster sind mit Input-/Operatorprüfung und Hash des Herkunftsberichts übernommen. Physikalische Grenzwerte wurden bei diesen Reparaturen nicht gelockert.

## 4. Reproduktion und nächste Arbeit

Aus dem Engine-Repository, Python 3.12 mit installierter Engine und Prüfextra `validation` (SciPy):

```sh
../WaveStudio/.venv/bin/python validation/quantitative_sources.py --policy validation/quantitative-v2-wide.json --output /tmp/quantitative-sources.json
../WaveStudio/.venv/bin/python validation/run_volume_references.py --policy validation/volume-reference-v1-followup.json --output /tmp/volume-reference.json
../WaveStudio/.venv/bin/python validation/run_references.py --only T02 --output /tmp/t02-diagnostic.json
```

Die ersten beiden Befehle liefern derzeit wegen offener Kriterien einen Fehlerstatus. CPU-Laufzeit und Speicherbedarf sind deutlich höher als bei den Kernprüfungen. SciPy wird ausdrücklich aus der Desktop-Paketierung ausgeschlossen; die Kern-Engine und die App benötigen diese neue Prüfabhängigkeit nicht.

Offen bleiben: Ursachen der Linien-Fenster-/Randabhängigkeit, passende endliche Messdefinition/Referenz, weitere Qualifikation der Linsenreferenz, unabhängige endliche Doppelspaltreferenz, dritte gezielte Engine-Rasterstufe für Spalt/Linse sowie getrennte Zeit-/Rand-/Geometrie-/Peak-Unsicherheit. Erst danach quantitative Preset-Aussagen und eine aus der jeweiligen Interferenzmessung abgeleitete Phasenforderung freigeben. Historische v1-PASS-Berichte bleiben unverändert.
