#!/bin/bash
#==========================
# Build-compile-run script
#==========================

#============ Constants ============#

PROJECT_DIR="$( cd -- "$(dirname -- "$0")" >/dev/null 2>&1 ; pwd -P )"
BUILD_DIR="${PROJECT_DIR}/build"
TEST_DIR="${BUILD_DIR}/tests"


#============ Actions ============#

# Help function
usage() {
less -Ps"Manual page setup\.sh line %lt (press h for help or q to quit)" << EOF
NAME
    setup.sh - config, build and run the module

SYNOPSIS
    sh setup.sh load_modules
                config [-D <build-type>]
                build [<target> ...]
                ready [-D <build-type>] [<target> ...]
                tests [--verbose|-v] [-m <module>|<pattern>]
                install
                uninstall
                clean

COMMANDS
    load_modules - load gcc, cmake, python and mpi modules (!!! only for Euler !!!)
    config    - Configure the project with cmake and the given options.
    build     - Compile the project using make with the cmake-generated files (!!! config must have been called first !!!).
    ready     - Configure and build the project (equivalent to calling 'config' and 'build').
    tests     - Run the project verification or a subset of it (!!! build must have been called first !!!)
    install   - Install the C++ dynamic library (requires Release build first).
    uninstall - Uninstalls the C++ dynamic library
    clean     - Deletes all the folders and files generated by cmake for a refresh.

OPTIONS
    -v, --verbose
        Show full output from Boost.test.

ARGUMENTS
    build-type      - CMake build type (can be one of: Release, Debug, Benchmark, Profile)
    target          - CMake target to compile (can be a benchmark or a test case).
    pattern         - Regular expression matching the names of the tests to run (run all by default).
    module          - Stem of the test file to run.

EOF
}

# Configure project with cmake
config() {
    if [ ! -d "$BUILD_DIR" ]; then
        mkdir $BUILD_DIR
    fi
    BUILD_TYPE=""
    while getopts "D:" opt; do
        case $opt in
            D)
                BUILD_TYPE="-D CMAKE_BUILD_TYPE=${OPTARG}"
                ;;
            ?)
                exit 1
                ;;
        esac
    done
    cmake -B "${BUILD_DIR}" ${BUILD_TYPE} "${PROJECT_DIR}"
}

# Compile project with cmake-generated Makefile
build() {
    if [ ! -d "$BUILD_DIR" ]; then
        echo "No 'build' directory found. The project must be configured before compiling."
        exit 1
    fi
    CLEAN_OPT=""
    while getopts "c" opt; do
        case $opt in
            c)
                CLEAN_OPT="--clean-first"
                ;;
            ?)
                exit 1
                ;;
        esac
    done
    shift $(( OPTIND - 1 ))
    TARGET_OPT=""
    if [ $# -gt 1 ]; then
        echo "Too many targets given."
        exit 1
    elif [ $# = 1 ]; then
        TARGET_OPT="--target $@"
    fi
    cmake --build "$BUILD_DIR" $CLEAN_OPT $TARGET_OPT
}

# Run project tests
tests() {
    if [ ! -d "${BUILD_DIR}" ]; then
        echo "No 'build' directory found. The project must be fully built before testing."
        exit 1
    fi
    NO_EXEC=true
    for item in "${TEST_DIR}"/*; do
        if [ ! -d $item -a -x $item ]; then
            NO_EXEC=false
        fi
    done
    if [ $NO_EXEC = true ]; then
        echo "There is no test executable. The tests must be compiled before testing."
        exit 1
    fi
    USE_MODULE=false
    PATTERN=""
    VERBOSITY=""
    while getopts "v-:m:" opt; do
        case $opt in
            v)
                VERBOSITY="-V"
                ;;
            m)
                USE_MODULE=true
                PATTERN=$OPTARG
                ;;
            -)
                if [ "${OPTARG}" = "verbose" ]; then
                    VERBOSITY="-V"
                else
                    echo "Unknown option: --${OPTARG}."
                    exit 1
                fi
                ;;
            ?)
                exit 1
                ;;
        esac
    done
    shift $(( OPTIND - 1 ))
    if [ $USE_MODULE = true ]; then
        NARGS=0
    else
        NARGS=1
    fi
    if [ $# -gt $NARGS ]; then
        echo "Too many arguments (${NARGS} expected, ${#} given)."
        exit 1
    elif [ $# = 1 ]; then
        PATTERN=$1
    fi
    (
        cd "$BUILD_DIR"
        if [ $USE_MODULE = true ]; then
            ctest ${VERBOSITY} -R "^${PATTERN}\."
        else
            if [ -n "$PATTERN" ]; then
                ctest ${VERBOSITY} -R "$PATTERN"
            else
                ctest ${VERBOSITY}
            fi
        fi
    )
}

# Clean files and folders
clean() {
    if [ -d "${BUILD_DIR}" ]; then
        rm -rf "${BUILD_DIR}"
    fi
    if [ -d "swht.egg-info" ]; then
        rm -rf "swht.egg-info"
    fi
}


#============ Action selection ============#

case $1 in
    "load_modules")
        module load gcc/8.2.0 boost/1.68.0 cmake/3.13.4 python/3.8.5 gmp/6.1.2
        ;;
    
    "ready")
        shift
        config $@
        if [[ $@ =~ .*-D.* ]]; then
            shift 2
        fi
        build $@
        ;;

    "config")
        shift
        config $@
        ;;
    
    "build")
        shift
        build $@
        ;;
    
    "tests")
        shift
        tests $@
        ;;
    
    "install")
        (
            cd "$BUILD_DIR";
            sudo make install
        )
        ;;

    "uninstall")
        (
            cd "$BUILD_DIR";
            sudo make uninstall
        )
        ;;

    "clean")
        clean
        ;;
    
    "help"|"-h")
        usage
        exit 0
        ;;
    
    *)
        echo "Unrecognized command '$0 $1'"
        usage
        exit 1
        ;;
esac
