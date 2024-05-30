import streamlit as st

st.write("# ΠΛΗΡΟΦΟΡΙΕΣ")

st.write("<p style='font-size:28px'>Η εφαρμογή επιλύει αλγορίθμους μηχανικής μάθησης και οπτικοποιεί dataset δεδομένων που υποβάλετε.</p>", unsafe_allow_html=True)

st.write("<br><p style='font-size:38px'>Οδηγός Χρήσης</p>", unsafe_allow_html=True)

st.write("<p style='font-size:31px'>2D οπτικοποίηση</p>", unsafe_allow_html=True)

st.write('Με το ποντίκι σας επιλέξτε το πλαίσο "browse files" και υποβάλετε το αρχείο που θέλετε.')

st.image("images/browse files.png")

st.write("Το αρχείο πρέπει να είναι μορφής CSV, XLSX ή XLS. Πρέπει να περιλαμβάνει μόνο αριθμητικές τιμές tabular data (SxF) διαστάσεων.")

st.write("<br><br>", unsafe_allow_html=True)

st.write("'Εχοντας υποβάλει το αρχείο, μπορείτε να επιλέξετε μέθοδο οπτικοποίησης PCA ή t-SNE.")

st.write("Εάν δημιουργήθεί εδώ σφάλμα του παρακάτω τύπου, το αρχείο που υποβάλατε δεν υποστηρίζεται από την εφαρμογή.")

st.image("images/error.png")

st.write("<br> <br>", unsafe_allow_html=True)

st.write("<p style='font-size:31px'>Αλγόριθμοι Κατηγοριοποίησης</p>", unsafe_allow_html=True)

st.write("Μπορείτε στο 2ο tab να χρησιμοποιήσετε τις λειτουργίες αλγορίθμων κατηγοριοποίησης. ")

st.image("images/classification.png")

st.write("Εδώ μπορείτε να επιλέξετε το πλήθος κοντινότερων γειτόνων και το πλήθος δέντρων για τους αλγόριθμους K-nearestNeighbors και RandomForest αντίστοιχα.  \n Οι αλγόριθμοι εκτελούνται αυτόματα και θα σας καταγράψουν αποτελέσματα απόδοσης, ακρίβειας και πίνακα σύγχυσης.  \n ")

st.write("<br> <br>", unsafe_allow_html=True)

st.write("<p style='font-size:31px'>Αλγόριθμοι Ομαδοποίησης</p>", unsafe_allow_html=True)

st.write("Στο 3ο tab μπορείτε να επιλέξετε τις λειτουργίες αλγορίθμων ομαδοποίησης K-means και Affinity Propagation.")

st.image("images/cluster.png")

st.write("Μπορείτε στον K-means αλγόριθμο να υποβάλετε τον αριθμό ομάδων που θα χρησιμοποιηθούν. O αλγόριθμος σας καταγράφει το Silhouette Score, την κατανομή των ομάδων και γράφος του αλγορίθμου.  \n  \nΟμοίως, στον Affinity Propagation αλγόριθμο μπορείτε να εισάγετε άλλη τιμή μέτρου προτίμησης από εκείνη που δίνεται, και θα καταγραφεί ο αριθμός ομάδων, το Silhouette Score, η κατανομή των ομάδων και γράφος του αλγορίθμου αντίστοιχα.")

st.write("<br> <br>", unsafe_allow_html=True)

st.write("<p style='font-size:41px'>Η ομάδα μας</p>", unsafe_allow_html=True)

st.write("Νικόλας Μοσχόβης - inf2021144 - Υπεύθυνος ανάπτυξης και σχεδιαστής εφαρμογής machine learning αλγορίθμων")

st.write("Παναγιώτης Ηλιόπουλος - inf2021060 - Υπεύθυνος σχεδιαστής διαγράμματος και κύκλου ζωής έκδοσης λογισμικού")

st.write("Κωνσταντίνος Γαρείος - inf2021036 - Υπεύθυνος αναφοράς, Info Tab, Docker/Github")
