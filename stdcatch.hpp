/* some small templates for quick std::exception catching
 */
#ifndef _STDCATCH_H_
#define _STDCATCH_H_

#define STD_CATCH_ERROR(e) \
  std::cerr << "Exception caught at " << __FILE__ << ": " << __LINE__ << ": " << e.what() << '\n'

#define STDCATCH \
  catch(std::exception &e) { STD_CATCH_ERROR(e); }

#define STDCATCHRETURN \
  catch(std::exception &e) { STD_CATCH_ERROR(e); return 1; }


#endif /* _STDCATCH_H_ */
