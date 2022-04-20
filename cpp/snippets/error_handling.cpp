/*
 * convert errors at subsystem boundaries
 *
 * consider designing a library with multiple error handling mechanisms
 * */

void foo(double);
void foo(double, int& error_code) noexcept;

/*
 * The fundamental problem is not about handling exceptions, but after an error is detected,
 * the program must remain in a well-defined state
 *
 * as a minimum, no resources must be leaked -- basic error safety guarantee
 *
 * ideally, operation which caused error is undone -- strong error safety guarantee
 * */
