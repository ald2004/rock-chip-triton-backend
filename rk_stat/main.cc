#include <rknn_api.h>
#include <iostream>
#ifdef _WIN32
// suppress the min and max definitions in Windef.h.
#define NOMINMAX
#include <Windows.h>
#else
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#endif
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>
#include <exception>

#define TRITON_ENABLE_LOGGING
namespace triton { namespace common {

// A log message.
class LogMessage {
 public:
  // Log levels.
  enum Level { kERROR = 0, kWARNING = 1, kINFO = 2 };

  LogMessage(const char* file, int line, uint32_t level);
  ~LogMessage();

  std::stringstream& stream() { return stream_; }

 private:
  static const std::vector<char> level_name_;
  std::stringstream stream_;
};

// Global logger for messages. Controls how log messages are reported.
class Logger {
 public:
  enum class Format { kDEFAULT, kISO8601 };

  Logger();

  // Is a log level enabled.
  bool IsEnabled(LogMessage::Level level) const { return enables_[level]; }

  // Set enable for a log Level.
  void SetEnabled(LogMessage::Level level, bool enable)
  {
    enables_[level] = enable;
  }

  // Get the current verbose logging level.
  uint32_t VerboseLevel() const { return vlevel_; }

  // Set the current verbose logging level.
  void SetVerboseLevel(uint32_t vlevel) { vlevel_ = vlevel; }

  // Get the logging format.
  Format LogFormat() { return format_; }

  // Set the logging format.
  void SetLogFormat(Format format) { format_ = format; }

  // Log a message.
  void Log(const std::string& msg);

  // Flush the log.
  void Flush();

 private:
  std::vector<bool> enables_;
  uint32_t vlevel_;
  Format format_;
  std::mutex mutex_;
};

// extern Logger gLogger_;

#define LOG_ENABLE_INFO(E)             \
  triton::common::gLogger_.SetEnabled( \
      triton::common::LogMessage::Level::kINFO, (E))
#define LOG_ENABLE_WARNING(E)          \
  triton::common::gLogger_.SetEnabled( \
      triton::common::LogMessage::Level::kWARNING, (E))
#define LOG_ENABLE_ERROR(E)            \
  triton::common::gLogger_.SetEnabled( \
      triton::common::LogMessage::Level::kERROR, (E))
#define LOG_SET_VERBOSE(L)                  \
  triton::common::gLogger_.SetVerboseLevel( \
      static_cast<uint32_t>(std::max(0, (L))))
#define LOG_SET_FORMAT(F) \
  triton::common::gLogger_.SetLogFormat((F))

#ifdef TRITON_ENABLE_LOGGING

#define LOG_INFO_IS_ON \
  triton::common::gLogger_.IsEnabled(triton::common::LogMessage::Level::kINFO)
#define LOG_WARNING_IS_ON             \
  triton::common::gLogger_.IsEnabled( \
      triton::common::LogMessage::Level::kWARNING)
#define LOG_ERROR_IS_ON \
  triton::common::gLogger_.IsEnabled(triton::common::LogMessage::Level::kERROR)
#define LOG_VERBOSE_IS_ON(L) (triton::common::gLogger_.VerboseLevel() >= (L))

#else

// If logging is disabled, define macro to be false to avoid further evaluation
#define LOG_INFO_IS_ON false
#define LOG_WARNING_IS_ON false
#define LOG_ERROR_IS_ON false
#define LOG_VERBOSE_IS_ON(L) false

#endif  // TRITON_ENABLE_LOGGING

// Macros that use explicitly given filename and line number.
#define LOG_INFO_FL(FN, LN)                                      \
  if (LOG_INFO_IS_ON)                                            \
  triton::common::LogMessage(                                    \
      (char*)(FN), LN, triton::common::LogMessage::Level::kINFO) \
      .stream()
#define LOG_WARNING_FL(FN, LN)                                      \
  if (LOG_WARNING_IS_ON)                                            \
  triton::common::LogMessage(                                       \
      (char*)(FN), LN, triton::common::LogMessage::Level::kWARNING) \
      .stream()
#define LOG_ERROR_FL(FN, LN)                                      \
  if (LOG_ERROR_IS_ON)                                            \
  triton::common::LogMessage(                                     \
      (char*)(FN), LN, triton::common::LogMessage::Level::kERROR) \
      .stream()
#define LOG_VERBOSE_FL(L, FN, LN)                                \
  if (LOG_VERBOSE_IS_ON(L))                                      \
  triton::common::LogMessage(                                    \
      (char*)(FN), LN, triton::common::LogMessage::Level::kINFO) \
      .stream()

// Macros that use current filename and line number.
#define LOG_INFO LOG_INFO_FL(__FILE__, __LINE__)
#define LOG_WARNING LOG_WARNING_FL(__FILE__, __LINE__)
#define LOG_ERROR LOG_ERROR_FL(__FILE__, __LINE__)
#define LOG_VERBOSE(L) LOG_VERBOSE_FL(L, __FILE__, __LINE__)


#define LOG_STATUS_ERROR(X, MSG)                         \
  do {                                                   \
    const Status& status__ = (X);                        \
    if (!status__.IsOk()) {                              \
      LOG_ERROR << (MSG) << ": " << status__.AsString(); \
    }                                                    \
  } while (false)

#define LOG_TRITONSERVER_ERROR(X, MSG)                                  \
  do {                                                                  \
    TRITONSERVER_Error* err__ = (X);                                    \
    if (err__ != nullptr) {                                             \
      LOG_ERROR << (MSG) << ": " << TRITONSERVER_ErrorCodeString(err__) \
                << " - " << TRITONSERVER_ErrorMessage(err__);           \
      TRITONSERVER_ErrorDelete(err__);                                  \
    }                                                                   \
  } while (false)

#define LOG_FLUSH triton::common::gLogger_.Flush()

}}  // namespace boetriton::common


namespace triton { namespace common {

Logger gLogger_;

Logger::Logger() : enables_{true, true, true}, vlevel_(0), format_(Format::kISO8601) {}

void
Logger::Log(const std::string& msg)
{
  const std::lock_guard<std::mutex> lock(mutex_);
  std::cerr << msg << std::endl;
}

void
Logger::Flush()
{
  std::cerr << std::flush;
}


const std::vector<char> LogMessage::level_name_{'E', 'W', 'I'};

LogMessage::LogMessage(const char* file, int line, uint32_t level)
{
  std::string path(file);
  size_t pos = path.rfind('/');
  if (pos != std::string::npos) {
    path = path.substr(pos + 1, std::string::npos);
  }

  // 'L' below is placeholder for showing log level
  switch (gLogger_.LogFormat())
  {
  case Logger::Format::kDEFAULT: {
    // LMMDD hh:mm:ss.ssssss
#ifdef _WIN32
    SYSTEMTIME system_time;
    GetSystemTime(&system_time);
    stream_ << level_name_[std::min(level, (uint32_t)Level::kINFO)]
            << std::setfill('0') << std::setw(2) << system_time.wMonth
            << std::setw(2) << system_time.wDay << ' ' << std::setw(2)
            << system_time.wHour << ':' << std::setw(2) << system_time.wMinute
            << ':' << std::setw(2) << system_time.wSecond << '.' << std::setw(6)
            << system_time.wMilliseconds * 1000 << ' '
            << static_cast<uint32_t>(GetCurrentProcessId()) << ' ' << path << ':'
            << line << "] ";
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    struct tm tm_time;
    gmtime_r(((time_t*)&(tv.tv_sec)), &tm_time);
    stream_ << level_name_[std::min(level, (uint32_t)Level::kINFO)]
            << std::setfill('0') << std::setw(2) << (tm_time.tm_mon + 1)
            << std::setw(2) << tm_time.tm_mday << ' ' << std::setw(2)
            << tm_time.tm_hour << ':' << std::setw(2) << tm_time.tm_min << ':'
            << std::setw(2) << tm_time.tm_sec << '.' << std::setw(6) << tv.tv_usec
            << ' ' << static_cast<uint32_t>(getpid()) << ' ' << path << ':'
            << line << "] ";
#endif
    break;
  }
  case Logger::Format::kISO8601: {
    // YYYY-MM-DDThh:mm:ssZ L
#ifdef _WIN32
    SYSTEMTIME system_time;
    GetSystemTime(&system_time);
    stream_ << system_time.wYear << '-'
            << std::setfill('0') << std::setw(2) << system_time.wMonth << '-'
            << std::setw(2) << system_time.wDay << 'T' << std::setw(2)
            << system_time.wHour << ':' << std::setw(2) << system_time.wMinute
            << ':' << std::setw(2) << system_time.wSecond << "Z "
            << level_name_[std::min(level, (uint32_t)Level::kINFO)] << ' '
            << static_cast<uint32_t>(GetCurrentProcessId()) << ' ' << path << ':'
            << line << "] ";
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    struct tm tm_time;
    gmtime_r(((time_t*)&(tv.tv_sec)), &tm_time);
    stream_ << (tm_time.tm_year + 1900) << '-'
            << std::setfill('0') << std::setw(2) << (tm_time.tm_mon + 1) << '-'
            << std::setw(2) << tm_time.tm_mday << 'T' << std::setw(2)
            << tm_time.tm_hour << ':' << std::setw(2) << tm_time.tm_min << ':'
            << std::setw(2) << tm_time.tm_sec << "Z "
            << level_name_[std::min(level, (uint32_t)Level::kINFO)] << ' '
            << static_cast<uint32_t>(getpid()) << ' ' << path << ':'
            << line << "] ";
#endif
    break;
  }
  }
}

LogMessage::~LogMessage()
{
  gLogger_.Log(stream_.str());
}

}}  // namespace boetriton::common

struct TRITONSERVER_Error;

#define LOG_MESSAGE(LEVEL, MSG)                                  \
  do {                                                           \
        TRITONSERVER_LogMessage(LEVEL, __FILE__, __LINE__, MSG), \
        ("failed to log message: ");                            \
  } while (false)

typedef enum TRITONSERVER_loglevel_enum {
  TRITONSERVER_LOG_INFO,
  TRITONSERVER_LOG_WARN,
  TRITONSERVER_LOG_ERROR,
  TRITONSERVER_LOG_VERBOSE
} TRITONSERVER_LogLevel;

TRITONSERVER_Error*
TRITONSERVER_LogMessage(
    TRITONSERVER_LogLevel level, const char* filename, const int line,const char* msg){
  switch (level) {
    case TRITONSERVER_LOG_INFO:
      LOG_INFO_FL(filename, line) << msg;
      return nullptr;
    case TRITONSERVER_LOG_WARN:
      LOG_WARNING_FL(filename, line) << msg;
      return nullptr;
    case TRITONSERVER_LOG_ERROR:
      LOG_ERROR_FL(filename, line) << msg;
      return nullptr;
    case TRITONSERVER_LOG_VERBOSE:
      LOG_VERBOSE_FL(1, filename, line) << msg;
      return nullptr;
    default:
      return nullptr;
  }
}

const char *getBuild() { //Get current architecture, detectx nearly every architecture. Coded by Freak

        #if defined(__x86_64__) || defined(_M_X64)
        return "x86_64";
        #elif defined(i386) || defined(__i386__) || defined(__i386) || defined(_M_IX86)
        return "x86_32";
        #elif defined(__ARM_ARCH_2__)
        return "ARM2";
        #elif defined(__ARM_ARCH_3__) || defined(__ARM_ARCH_3M__)
        return "ARM3";
        #elif defined(__ARM_ARCH_4T__) || defined(__TARGET_ARM_4T)
        return "ARM4T";
        #elif defined(__ARM_ARCH_5_) || defined(__ARM_ARCH_5E_)
        return "ARM5"
        #elif defined(__ARM_ARCH_6T2_) || defined(__ARM_ARCH_6T2_)
        return "ARM6T2";
        #elif defined(__ARM_ARCH_6__) || defined(__ARM_ARCH_6J__) || defined(__ARM_ARCH_6K__) || defined(__ARM_ARCH_6Z__) || defined(__ARM_ARCH_6ZK__)
        return "ARM6";
        #elif defined(__ARM_ARCH_7__) || defined(__ARM_ARCH_7A__) || defined(__ARM_ARCH_7R__) || defined(__ARM_ARCH_7M__) || defined(__ARM_ARCH_7S__)
        return "ARM7";
        #elif defined(__ARM_ARCH_7A__) || defined(__ARM_ARCH_7R__) || defined(__ARM_ARCH_7M__) || defined(__ARM_ARCH_7S__)
        return "ARM7A";
        #elif defined(__ARM_ARCH_7R__) || defined(__ARM_ARCH_7M__) || defined(__ARM_ARCH_7S__)
        return "ARM7R";
        #elif defined(__ARM_ARCH_7M__)
        return "ARM7M";
        #elif defined(__ARM_ARCH_7S__)
        return "ARM7S";
        #elif defined(__aarch64__) || defined(_M_ARM64)
        return "ARM64";
        #elif defined(mips) || defined(__mips__) || defined(__mips)
        return "MIPS";
        #elif defined(__sh__)
        return "SUPERH";
        #elif defined(__powerpc) || defined(__powerpc__) || defined(__powerpc64__) || defined(__POWERPC__) || defined(__ppc__) || defined(__PPC__) || defined(_ARCH_PPC)
        return "POWERPC";
        #elif defined(__PPC64__) || defined(__ppc64__) || defined(_ARCH_PPC64)
        return "POWERPC64";
        #elif defined(__sparc__) || defined(__sparc)
        return "SPARC";
        #elif defined(__m68k__)
        return "M68K";
        #else
        return "UNKNOWN";
        #endif
    }
int main(int argc,char* argv[]){
    rknn_context ctx;
    rknn_sdk_version version;
    rknn_mem_size mem_size;
    std::string deviceArch(getBuild());
    if(deviceArch.compare("ARM64")){ // rk3588
      

    }else{  //rv1126

    }
    try
    {
        std::string modelPath;
        if(argc >1){
            modelPath=std::move(argv[1]);
        }else
            modelPath=std::move("model.rknn");
        int ret =-1;
        ret= rknn_init(&ctx, (void*)modelPath.c_str(), 0, 0 , NULL); 
        if(ret<0)
           throw std::exception();
        ret = rknn_query(ctx, RKNN_QUERY_MEM_SIZE, &mem_size, sizeof(mem_size));
        if(ret<0)
           throw std::exception();
        LOG_MESSAGE(TRITONSERVER_LOG_INFO,(std::string("rk_stat model :")+modelPath+
            std::string("\n rknn_mem_size : \n\t total_weight_size : ")+
            std::to_string(mem_size.total_weight_size)+std::string("\n\t total_internal_size : ")+
            std::to_string(mem_size.total_internal_size)).c_str());
    }
    catch(const std::exception& e)
    {
        LOG_MESSAGE(TRITONSERVER_LOG_ERROR,(std::string("rknn_init or rknn_query error!")).c_str());
    }
    return 0;
    
    

}