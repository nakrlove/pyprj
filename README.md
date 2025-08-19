# PyPrj Django 프로젝트 설정 가이드

이 문서는 `pyprj` Django 프로젝트를 로컬 환경에 설정하고 실행하기 위한 기본 작업 순서를 정리한 가이드입니다.

---

## 1️⃣ Git 저장소 복제

먼저 GitHub 저장소를 로컬로 복제합니다.

```bash
git clone "https://github.com/nakrlove/pyprj.git"

가상환경에서 실행
pip install django

python manage.py makemigrations bbs
python manage.py migrate
