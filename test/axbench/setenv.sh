#!/bin/bash -x


taffo_setenv_find()
{
  cd $1
  BASEDIR=$(pwd)

  if [[ $2 == 'lib' ]]; then
    SOEXT="so"
    if [ $(uname -s) = "Darwin" ]; then
      SOEXT="dylib";
    fi
    FN="$3.$SOEXT"
  else
    FN="$3"
  fi

  PATH="$BASEDIR/$2/$FN"

  if [[ ! -e "$PATH" ]]; then
    echo "Cannot find $FN" > /dev/stderr
  else
    echo "Found $PATH" > /dev/stderr
    echo "$PATH"
  fi
}


SCRIPTPATH=$(dirname "$BASH_SOURCE")

if [[ $# < 1 ]]; then
  echo 'Usage: source setenv.sh /prefix/where/taffo/was/installed'
  echo 'Install taffo by running make install. The default prefix is /usr/local.'
else
  export TAFFOLIB=$(taffo_setenv_find $1 'lib' 'Taffo')
  # Pass-specific lib variables for backwards compatibility:
  export INITLIB=$TAFFOLIB
  export VRALIB=$TAFFOLIB
  export TUNERLIB=$TAFFOLIB
  export PASSLIB=$TAFFOLIB
  export ERRORLIB=$TAFFOLIB
  export INSTMIX=$(taffo_setenv_find $1 'bin' 'taffo-instmix')
  export TAFFO_MLFEAT=$(taffo_setenv_find $1 'bin' 'taffo-mlfeat')
  export TAFFO_FE=${SCRIPTPATH}/../ErrorAnalysis/FeedbackEstimator/taffo-fe.py
  export TAFFO_PE=${SCRIPTPATH}/../ErrorAnalysis/PerformanceEstimator/taffo-pe.py

  if [[ -z "$LLVM_DIR" ]]; then
    LLVM_DIR=$(llvm-config --prefix 2> /dev/null)
    if [[ $? -ne 0 ]]; then
      printf "*** WARNING ***\nCannot set LLVM_DIR using llvm-config\n"
    fi
  fi
  if [[ ! ( -z "$LLVM_DIR" ) ]]; then
    if [ $(uname -s) = "Darwin" ]; then
      # xcrun patches the command line parameters to clang to add the standard
      # include paths depending on where the currently active platform SDK is
      export CLANG="xcrun $LLVM_DIR/bin/clang"
      export CLANGXX="xcrun $LLVM_DIR/bin/clang++"
    fi
  fi
fi

unset -f taffo_setenv_find
unset -v SCRIPTPATH
