# installation


python -m venv venv

source venv/###/activate

pip install -r requirements_inf.txt

docker redis

python manage.py migrate

python manage.py runserver

python sender.py