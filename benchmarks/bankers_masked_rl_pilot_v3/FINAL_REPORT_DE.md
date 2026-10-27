# Deutscher Abschlussbericht — BankerShield Masked RL Pilot V3

**BANKERS_MASKED_RL_PILOT_V3_NO_LEARNING_SIGNAL**

1. **Repository und Identität.** Branch `research/bankers-masked-rl-pilot-v3`; Worktree `/Users/romanliessner/gitlab/gym_longicontrol_bankers_masked_rl_pilot_v3`; Basis `e4dd7cb0e4300cb9b66cf4b62cc39340247b0e40`; wissenschaftlicher Freeze `9f6894e0b2a2a8e3b3cc08e942b7b177c7657844`. Verifizierter Push-Remote ausschließlich `gym_longicontrol_origin` → `https://github.com/dynamik1703/gym_longicontrol.git`. Der Publikationscommit ist der nach Abschluss verifizierte Branch-HEAD; keine Force-Pushes.

2. **Historie.** Alle sechs früheren Klassifikationen, Pakete, Attempt-Ledger und Ergebnisse entsprechen unverändert der eingefrorenen Basis. V2 bleibt `BANKERS_MASKED_RL_PILOT_V2_RESOURCE_STOP` mit 18,985046 h / 2,042108 GiB. V3 ist ein prospektiv neuer Versuch, keine rückwirkende Budgetkorrektur.

3. **Methode.** Unveränderter V2-SAC: kausale 10D-Beobachtung; 21 native Aktionen; 256/256-ReLU; zwei Reward- und drei Cost-Critics; native Gamma `0.99**0.1`; Tau 0,005; unveränderte Lernraten und Koeffizienten. Reward = −1000 × signierte Schrittenergie in kWh; separate Speed-, Terminal- und Interventionskosten. RAW führt Vorschläge direkt aus. POST behält den Vorschlag im Replay und ersetzt durch den nächstgelegenen zertifizierten Gitterindex, kleinerer Index bei Gleichstand. PRE maskiert Sampling, Entropie und Bellman-Ziele. Beide Banker-Arme erhalten dieselbe unveränderte physikalische V4-Endzertifizierung. Kein Warmstart, Cloning, Progress-Bonus, Tuning oder Chunking.

4. **Ressourcenaufnahme.** Vor Einsicht in das Gate-Ergebnis eingefrorenes vollständiges Modell: 41.370,753270 s = 11,491876 h aktive Rechenzeit als konservative Wall-Raten-Projektion; 2.102.448.583,75 Bytes = 1,958058 GiB. Grenzen: 24 h Prozess-CPU / 3 GiB neue Artefakte. Vollständige Trainings-/Diagnostic-/Audit-/Checkpoint-/Recovery-/Sealing-Kosten sind enthalten. Zusätzlich dokumentiertes Extrem-Szenario mit wiederholten individuellen Wall-Maxima: 161,4145 h; es ist getrennt vom vorab festgelegten Aufnahmeverfahren. Siehe RESOURCE_PROTOCOL.md und RESOURCE_REPORT.md.

5. **Kaltstart und laufende Kosten.** Drei Wiederholungen × 128 Schritte × sieben synthetische Geschwindigkeitsbänder, ohne reservierte Performance-Aufgaben. CPU/Wall, Median/p95/p99/Maximum und Komponenten sind gespeichert. V2s RAW-Ausreißer: 124,179 ms Wall, 3,522 ms CPU. Im neuen Profil liegt RAWs größter Schritt bei 1,974 s Wall / 5,117 ms CPU, davon 1,962 s im Journal-Append mit dauerhafter I/O. Damit ist Wartezeit innerhalb der Evidenz-I/O belegt; die exakte Ursache des alten V2-Ereignisses ist rückwirkend nicht identifizierbar. Kein Messfehler behauptet, kein Sample entfernt. Ein vor dem Freeze erkannter MPC-Profilharness-TimeLimit-Fehler wurde mit Regressionstest repariert und sein fehlgeschlagener Job mit 998,382 s konservativ belastet. Keine wissenschaftliche Performance war geöffnet.

6. **Engineering/Preflight.** 153 vollständige Engineering-Tests, weitere fokussierte 13- und 15-Test-Läufe; der einmalige versiegelte Preflight der finalen Quelle bestand alle 155 Tests, null Failures/Errors. Darunter echte Prozessabbrüche, exakte Recovery, Vorgänger-/Envelope-Rekonstruktion, sämtliche Gitter-Gleichstände, Masken-/Bellman-/Entropieprüfungen, Signed Energy, Korruption, Source-/Ressourcen- und Access-Guards. Alle wissenschaftlichen Gates bestanden.

7. **Exakte Trainingszahlen.** Seed 11, unveränderter geordneter Stream SHA-256 `08e8e5b0f03dd7caae95b3f1175edaa22a29608cbc9a554fea6e3be3e24cfe3c`. RAW, POST und PRE jeweils genau 50.000 native Übergänge / 5.000 vollständige SAC-Updates; insgesamt 150.000 / 15.000. Null Updates bis einschließlich Übergang 10.000; ursprünglicher kumulativer Plan. Nur finale 50k-Checkpoints evaluiert.

8. **Finale Hashes.**

`BANKER_POST_SAC`: Actor SHA-256 `c40bd4f65abb1f47e86e8538aeda2fb951f5427ce77e5f4918e8d0ff748be1cb`; Checkpoint SHA-256 `7dd2a62e6768b7f209d2581be91fbb0ff95c423fa0c328346d82db7dcf609a29`.

`BANKER_PRE_SAC`: Actor SHA-256 `0d6a9683b6e21fbada4f079ef2804a92ae2f806d5b7ec9da06a02b3d53a26464`; Checkpoint SHA-256 `93cd8120fff5b6d11537b5310469abffb5f5dbede4b7c8f96b30c09e6a1901e4`.

`RAW_DISCRETE_SAC`: Actor SHA-256 `7832c01746ff1a7f8607ef99b4a7178ff3e6332c1d3c68974633b85de1ee7c8d`; Checkpoint SHA-256 `850c23734f870d77bbe8d99aa95e3b3f8764d807a0ec631c3681aed565848913`.

9. **Recovery/exakt einmal.** Ein neuer unveränderlicher Attempt pro Arm und Paar; exklusive Eigentümerschaft; dauerhaftes STARTED vor Reset; atomare volle Checkpoints. Alle Recovery-Tests einschließlich echter Prozessabbrüche bestanden. Im wissenschaftlichen Lauf war keine Recovery oder frische Wiederholung nötig. Je Arm genau 50.000 Replay-Slots unabhängig gelesen und geprüft.

10. **Diagnostic-Split.** `BANKERS_MASKED_RL_PILOT_DIAGNOSTIC_V3`; erster route-clean Policy-R-Zwölfer nach Scan ab 12100000, 16 Kandidaten geprüft, ausschließlich Metadaten-Schutzfilter. Ordered SHA-256 `e5fb92ae36eb9b57eb3be2ffe5e801b32ae9de425118fc32b7538d7f5a1ad5a4` vor der ersten Performance-Episode eingefroren. Genau 72 eindeutige Controller-/Task-Paare, jedes einmalig und unabhängig verifiziert.

11. **Feasibility je Controller.** Alle Rohnenner bleiben 12. Die folgende Tabelle gilt ausschließlich für diesen einen Seed und diese zwölf Routen.

| Controller | RSR /12 | Completion /12 | Deadline /12 | Speed /12 | Median Fortschritt m | Feasible Energie kWh (n) |
|---|---:|---:|---:|---:|---:|---:|
| ADAPTIVE_PRICE_MPC_V2 | 12 | 12 | 12 | 12 | 1000.165 | 0.167539 (12) |
| BANKERS_MASKED_CAUSAL60 | 12 | 12 | 12 | 12 | 1000.827 | 0.198182 (12) |
| BANKER_POST_SAC | 0 | 0 | 0 | 12 | 24.176 | nicht verfügbar (0) |
| BANKER_PRE_SAC | 0 | 0 | 0 | 12 | 29.042 | nicht verfügbar (0) |
| CAUSAL60 | 12 | 12 | 12 | 12 | 1000.980 | 0.196299 (12) |
| RAW_DISCRETE_SAC | 0 | 0 | 0 | 12 | 0.000 | nicht verfügbar (0) |

12. **Aufgabenpaarige Primärvergleiche.** PRE–RAW, PRE–POST und POST–RAW: jeweils beide fehlgeschlagen auf allen zwölf Aufgaben; keine einseitigen oder gemeinsamen RSR-Erfolge. Mittlere gepaarte Fortschrittsdifferenzen PRE–RAW +16,452 m, PRE–POST +9,864 m und POST–RAW +6,588 m. Das ist kein wirksames Absolvieren der 1000-m-Aufgabe und kein Überlegenheits-/Äquivalenzbeweis. Gemeinsam feasible Energievergleiche der Lerner: n=0, daher keine Energiedifferenz ausgewiesen.

13. **Actor und Maskierung.** Alle drei Lerner: 21 zulässige Aktionen an jedem besuchten Diagnostic-Zustand, keine Proposal-/Execution-Abweichung und null POST-Korrekturen. PRE-Entropie im Mittel 3,03333 nats bei maximal ln(21)=3,04452, also nahezu maximale kategoriale Entropie; die deterministische Argmax-Auswahl ergibt dennoch ineffektiven Fortschritt. Kein beobachteter zusätzlicher aktiver PRE-Maskeneingriff im Diagnostic. Vollständige Proposal-/Execution-Verteilungen, Maskenhistogramme und finite Optimizer-/Critic-/Multiplier-Statistiken stehen in ACTION_AND_MASK_STATISTICS.md, OPTIMIZATION_STABILITY.md und detailed_analysis.json.

14. **Feasible-only Energie und Referenzen.** Kein Lerner lieferte eine feasible Episode; seine Energie wird deshalb nicht als effizienter Fahrbetrieb verglichen. Alle drei physikalischen Referenzen waren 12/12 feasible. Mittelwerte: CAUSAL60 0,196299 kWh; BANKERS_MASKED_CAUSAL60 0,198182 kWh; AdaptivePrice MPC V2 0,167539 kWh. Gemessene mittlere Reisezeiten 85,125 / 85,192 / 101,783 s. Masked-CAUSAL60 korrigierte 87,929 % der nativen Vorschläge; MPC meldete 0,6468 % physikalische Overrides. Diese unterschiedlichen Eingriffstypen werden getrennt beschrieben.

15. **Exakte Fehleridentitäten.** Jeder Lerner scheiterte auf denselben zwölf numerischen Tasks an INCOMPLETE und DEADLINE: 12100000, 12100001, 12100003, 12100004, 12100005, 12100006, 12100007, 12100009, 12100010, 12100011, 12100014, 12100015. Keine Speed-Verletzung; keine fehlgeschlagene Referenzaufgabe. Vollständige Route-/Full-Task-Hashes und gepaarte Zeilen: split_ledger.json und detailed_analysis.json.

16. **Beobachtete Ressourcen.** Der komplette Trainings-/Diagnostic-Prozess einschließlich Start und Kindprozessen: 3.097,14 CPU-s / 7.052,03 Wall-s. Erfolgreiches Profil: 706,07 CPU-s / 959,27 Wall-s. Versiegelter Preflight: 530,19 CPU-s / 838,64 Wall-s. Engineering/fokussierte Tests: 583,335901 gemessene CPU-s. Deskriptive Aufbereitung: 115,72 CPU-s / 129,14 Wall-s; ihr ergänzender JSON-Export benötigte ausschließlich eine Schlüsseltyp-Korrektur aus bereits gespeicherten Aggregaten, dokumentiert in REPORT_OUTPUT_READBACK_NOTE.md. Die CPU des ersten fehlgeschlagenen Profiljobs ist unbekannt und wird mit 998,382 s oberer Schranke separat belastet; keine erfundene exakte Gesamt-CPU. Wissenschaftlicher Ledger bei Abschluss 6.028,125 s konservativ belastet, inklusive vorheriger Arbeit/Reserven. Vor diesem Publikationsnachtrag 297.972.368 gespeicherte Bytes (~0,278 GiB); finale Byte-/Hash-Prüfung bei Veröffentlichung. Siehe OBSERVED_RESOURCE_ACCOUNTING_DE.md.

17. **Unabhängige native Prüfung.** Alle 150.000 nativen Übergänge und Replay-Insertionen zweimal im Kontroll-/Diskpfad geprüft; POST/PRE je 50.000 ausgewählte physikalische Aktionen vollständig zertifiziert und 1.000 aktuelle/nachfolgende Gitterzustände (21.000 Kandidaten) auf der eingefrorenen deterministischen Trainingsstichprobe rekonstruiert. Im Diagnostic vollständige native Prüfung und sämtliche relevanten Kandidatengitter; finale Actor-Law gegen den unveränderten Checkpoint geprüft. 1.393 Einträge des ursprünglichen Abschlussmanifests erfolgreich zurückgelesen; keine historische Datei verändert.

18. **Prospektiver Vollmaßstab.** Für PRE allein, 3 Seeds ×300k: aus tatsächlichen Pilotkosten mit eingefrorener 2×-Reserve 5,582 CPU-h / 12,043 Wall-h; alle drei Treatments, je 3×300k: 14,365 CPU-h / 34,797 Wall-h. Das konservative synthetische Geschwindigkeits-/Kostenmodell ergibt separat 50,313 h / 104,615 h. Online-Masken, Optimierung, Evidenz, unabhängige Audits, Checkpoints und Referenzpanels sind enthalten; Reserve und Modelldifferenzen bleiben sichtbar. Diese linearen Kosten-Szenarien der schwachen Pilotpolicies garantieren weder Laufzeit noch Leistung besser fahrender 300k-Policies. V2s 110/230-h-Projektion bleibt unverändert.

19. **Wissenschaftliche Klassifikation.** `BANKERS_MASKED_RL_PILOT_V3_NO_LEARNING_SIGNAL`: gültige und vollständig ausgeführte Studie; keine Resource-/Integrity-Abbruchklassifikation und kein Safety-Counterexample.

20. **Forschungsfrage und weitere Trainings.** Unter diesem Budget ist kein nützlicher Lernvorteil des physikalischen Pre-Shields erkennbar. PRE gewinnt nur wenige Meter, absolviert aber ebenso wie RAW/POST keine Aufgabe. Die Speed-Compliance ist hier auch für RAW 12/12; PRE maskiert im Diagnostic keine Aktion aus. Sie belegt daher weder erfolgreiche gelernte Kontrolle noch Energieeffizienz und lässt sich nicht als zusätzlicher aktiver Shield-Nutzen ausgeben. Ein Seed und zwölf Routen liefern keine allgemeine Widerlegung der Methode und keine 300k-Leistungsaussage. Automatische größere Trainings sind durch dieses Ergebnis nicht gerechtfertigt oder autorisiert.

21. **Final Validation.** Genau null finale Validation-Performance-Episoden.

22. **Paper.** Seeds 4000–4017 blieben SEALED_UNINSPECTED; keine Instanziierung, kein neues Fingerprinting, keine Performance-Inspektion dieser Tasks.

23. **Publikation.** Alle Quellen, Konfigurationen, Checkpoints, Evidenzen und Berichte werden auf dem genannten Branch ausschließlich an den verifizierten LongiControl-Remote publiziert. Nach dem Publikationscommit werden sauberer Worktree und Gleichheit von lokalem HEAD, Upstream und GitHub-Branch read-only geprüft und im Abschluss der Sitzung mit dem tatsächlichen Commit gemeldet. Kein Force-Push, kein automatischer Scale-up.
