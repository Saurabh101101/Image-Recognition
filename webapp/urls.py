from django.contrib import admin
from django.urls import path
from webapp import views
from django.conf.urls.static import static
from django.conf import settings
handler404 = 'webapp.views.entry_not_found'
urlpatterns =[
    path('',views.register, name='register'),
    path('predictImage',views.predictImage,name="predictImage"),
    path('about',views.about,name="about"),
    path('home',views.home,name="home"),
    path('register',views.register,name="register"),
    path('entry_not_found',views.entry_not_found,name="entry_not_found"),
    # path('viewdatabase',views.viewdatabase,name="viewdatabase"),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)