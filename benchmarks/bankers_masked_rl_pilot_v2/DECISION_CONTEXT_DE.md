# Konkretes No-Go nach vollständiger Ressourcenmessung

**BANKERS_MASKED_RL_PILOT_V2_RESOURCE_STOP**. Beide prospektiven Grenzen sind verletzt:
18.985046 h >8 h und
2.042108 GiB >2 GiB.

Die Projektion verwendet die vorab festgelegten Maximalraten je Arm und
Geschwindigkeitsband, sämtliche Sicherheits-/Evidenz-/Optimierungskosten und
Faktor 2. Sie ist eine konservative aktive CPU-Obergrenze auf Basis gemessener
Wall-Raten einschließlich I/O, keine tatsächlich verbrauchte Trainingszeit.
Insbesondere darf der doppelte Worst-Case-Aufwand nicht nachträglich als
gemessener CPU-Verbrauch ausgegeben werden.

Der vollständige Engineering-Durchgang bestand 147 Tests, einschließlich echter
Prozessabbrüche und aller neunzehn Gitter-Gleichstände. Letzter Engineering-Lauf:
510.107697 s tatsächlich gemessene CPU und
773.162815 s Wall-Zeit. Die vollständige Ressourcenprobe verbrauchte
23.529807 s CPU. Frühere Engineering-CPU wurde nicht einzeln
gemessen; ihre Wall-Zeiten sind konservativ separat eingerechnet, siehe
resource_accounting.json. Es existiert kein gemessener Trainings- oder Diagnostic-Verbrauch.

Wissenschaftliche Zähler je RAW/POST/PRE: **0/50000 native Transitionen und
0/5000 SAC-Update-Iterationen**; insgesamt 0/150000 und 0/15000.
Keine wissenschaftlichen Attempts, keine finalen Actor-/Checkpoint-Hashes,
kein Diagnostic-Panel und kein post-Freeze-Preflight. Die archivierende finale
Git-Version ist keine Freigabe durch einen erfolgreichen wissenschaftlichen Freeze.

Prospektiv: Banker PRE ×3 Seeds ×300k =
110.015658 h; alle drei Methoden ×3 Seeds ×300k =
229.796748 h.
Damit ist unter diesem expliziten Kostenmodell weder eine <=24h-Vollstudie
begründet noch eine Vollstudie wissenschaftlich gerechtfertigt. Die Lernbarkeitsfrage
bleibt mangels wissenschaftlicher Trainings-/Diagnostic-Daten offen.

Keine Matrixverkleinerung, kein Auslassen von Zertifikaten oder unabhängiger
Verifikation und keine nachträgliche Änderung des Ressourcenfaktors erfolgten.
BankerShield V1–V4, Pilot V1 und deren Incidents bleiben unverändert;
historische Validation bleibt unangetastet und Paper 4000–4017 SEALED_UNINSPECTED.
