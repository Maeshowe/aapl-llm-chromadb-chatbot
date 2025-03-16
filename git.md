# 🚀 GitHub Használati Útmutató (Git)

Ez az útmutató három részre van bontva, hogy kezdőként könnyen indulj, középhaladóként fejlődhess, és profiként is kihasználhasd a GitHub összes lehetőségét.

---

## 🌱 Kezdő szint

### 📌 Repository klónozása
Egy meglévő repository letöltése:
```bash
git clone https://github.com/felhasznalonev/repo-neve.git
```

### 📌 Új fájlok hozzáadása
```bash
git add <fájlnév>          # egy adott fájl hozzáadása
git add .                 # összes módosított fájl hozzáadása
```

### 📌 Commit készítése (mentés verziókövetésbe)
```bash
git commit -m "Commit üzenet"
```

### 📌 Változások feltöltése GitHubra
```bash
git push origin main
```

### 📌 GitHub frissítések letöltése
```bash
git pull origin main
```

---

## 🌿 Haladó szint

### 📌 Branch kezelés
Új branch létrehozása:
```bash
git branch feature-branch
```

Branch váltás:
```bash
git checkout feature-branch
```

Branch létrehozása és váltás azonnal:
```bash
git checkout -b feature-branch
```

Branch-ek listázása:
```bash
git branch
```

Branch feltöltése GitHubra:
```bash
git push -u origin feature-branch
```

Branch merge-elése:
```bash
git checkout main
git merge feature-branch
```

### 📌 Repository frissítése
Távoli módosítások letöltése:
```bash
git pull origin main
```

---

## 🌳 Haladó szint

### 📌 Git stash használata (változások ideiglenes félretétele)
Elmenti a jelenlegi módosításokat:
```bash
git stash
```

Visszatöltése:
```bash
git stash pop
```

### 📌 Merge konfliktus kezelése
Konfliktus esetén manuálisan javítsd a fájlt, majd:
```bash
git add <javított_fájl>
git commit -m "Konfliktus megoldva"
```

### 📌 Commit visszavonása
Utolsó commit visszavonása (változások megtartása mellett):
```bash
git reset --soft HEAD~1
```

Utolsó commit törlése véglegesen:
```bash
git reset --hard HEAD~1
```

### 📌 Commitok összevonása (Squash)
Több commit egyesítése eggyé:
```bash
git rebase -i HEAD~3
```

---

## 🌟 Professzionális szint

### 📌 Távoli repository kezelése
Távoli repository URL-jének változtatása:
```bash
git remote set-url origin <új-url>
```

Több távoli repository kezelése:
```bash
git remote add upstream <másik-url>
```

### 📌 Git alias-ok használata
Alias létrehozása gyakori parancsokhoz:
```bash
git config --global alias.st status
git config --global alias.ci commit
git config --global alias.br branch
git config --global alias.co checkout
```
Ezután használhatod például:
```bash
git st
git ci -m "üzenet"
```

### 📌 Git Workflows
- **Feature branches workflow:** Új funkciókat külön branch-ben dolgozol ki, majd merge-elsz.
- **Git Flow** munkafolyamat: Main, develop és feature ágak kezelése struktúráltan.

### 📌 GitHub Actions (CI/CD)
Automatikus tesztelés és deployment konfigurálása a GitHub repository-ban (`.github/workflows/*.yml` fájlokban).

---

## 🔖 Hasznos parancsok összegzése
- Repository állapota:
  ```bash
  git status
```
- Commit előzmények áttekintése:
```bash
git log --oneline --graph
```
- Grafikus áttekintés commitokról:
```bash
git log --oneline --graph --decorate --all
```

---

✅ Most már készen állsz arra, hogy professzionálisan kezeld a projektjeidet GitHub-on!

