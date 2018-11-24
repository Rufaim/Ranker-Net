TEMPLATE = lib
TARGET = rankernet
CONFIG += staticlib c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
    ranker/ranker_utils.cpp \
    ranker/dense.cpp \
    ranker/nalu.cpp \
    object_ranker.cpp

HEADERS += \
    json/json.hpp \
    ranker/ranker_utils.h \
    ranker/dense.h \
    ranker/layer.h \
    ranker/nalu.h \
    object_ranker.h
