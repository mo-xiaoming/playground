#include <cstdint>

/* std::uint8_t alikes generate much longer assembly code, because
 * they can be alias to *dst*
 * Any other non byte/char types generate shorter code
 * So, strong types are GOOD*/
void foo(std::uint8_t const *src, std::uint32_t *dst, int s) {
    for (auto i=0; i<s; ++i) {
        dst[i] = src[i];
    }
}
