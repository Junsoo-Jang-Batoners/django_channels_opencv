# installation


python -m venv venv

source venv/###/activate

pip install -r requirements_inf.txt

docker redis

python manage.py migrate

python manage.py runserver

python sender.py

gdrive down
https://drive.google.com/file/d/1eepjg7H3e-SCsDVBr9Qqb6U9FXCOlyES/view?usp=sharing
https://drive.google.com/file/d/1yz2skU5tuIi3-Zx85N1NXeYIwxsilcqs/view?usp=sharing

cd inference

python -m signjoey configs/sing_dcba_inf.yaml
